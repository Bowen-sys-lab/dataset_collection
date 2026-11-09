"""Load quantized AeroVLA once and answer bounded local image requests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exchange', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--max-requests', type=int, required=True)
    args = parser.parse_args()
    from scripts.model_worker_protocol import write_json, serve_requests
    args.exchange.mkdir(exist_ok=False)
    os.environ['HF_HOME'] = str(args.exchange.resolve()/'hf-cache')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    ready = {'status': 'FAIL', 'input_mode': 'duplicate_front_diagnostic', 'pid': os.getpid(),
             'model_load_count': 0, 'worker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor
        from peft import PeftModel
        from scripts.aerovla_offline_probe import quantization_config
        ready['model_input_files'] = {}
        for root in (args.base, args.adapter):
            for path in sorted(root.rglob('*')):
                if path.is_file() and '.cache' not in path.parts:
                    with path.open('rb') as stream:
                        ready['model_input_files'][str(path.resolve())] = hashlib.file_digest(stream, 'sha256').hexdigest()
        tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=True)
        processor = AutoImageProcessor.from_pretrained(args.base, local_files_only=True, trust_remote_code=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = AutoModelForVision2Seq.from_pretrained(args.base, local_files_only=True, trust_remote_code=True,
                    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={'': 0},
                    attn_implementation='eager', quantization_config=quantization_config())
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
        model.eval()
        torch.cuda.synchronize()
        ready.update(status='PASS', model_load_count=1, load_seconds=time.perf_counter()-started,
                     gpu=torch.cuda.get_device_name(), peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write_json(args.exchange/'ready.json', ready)

        def predict(request, folder):
            with Image.open(request['observation']['rgb_path']) as source:
                front = source.convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)
            mosaic = Image.new('RGB', (224, 448))
            mosaic.paste(front, (0, 0))
            mosaic.paste(front, (0, 224))
            mosaic.save(folder/'input_mosaic.png')
            prompt = f'<image>\n{request["instruction"]}\nAction: '
            inputs = tokenizer(prompt, return_tensors='pt')
            inputs['pixel_values'] = processor(images=mosaic, return_tensors='pt')['pixel_values'].to(torch.bfloat16)
            inputs = {key: value.to('cuda:0') for key, value in inputs.items()}
            started = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=20, do_sample=False,
                                        eos_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            generated = output[0, inputs['input_ids'].shape[1]:]
            return {'raw_output': tokenizer.decode(generated, skip_special_tokens=False),
                    'generated_token_ids': generated.tolist(), 'prompt': prompt,
                    'input_mode': 'duplicate_front_diagnostic', 'worker_pid': os.getpid(),
                    'inference_seconds': time.perf_counter()-started,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                    'mosaic_sha256': hashlib.sha256((folder/'input_mosaic.png').read_bytes()).hexdigest()}

        serve_requests(args.exchange, predict, max_requests=args.max_requests)
        return 0
    except Exception as error:
        if not (args.exchange/'ready.json').exists():
            ready['error'] = f'{type(error).__name__}: {error}'
            write_json(args.exchange/'ready.json', ready)
        (args.exchange/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
