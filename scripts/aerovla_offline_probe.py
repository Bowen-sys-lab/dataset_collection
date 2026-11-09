"""Run one explicitly labelled duplicate-front AeroVLA diagnostic without flying."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import traceback


def parse_action(text):
    match = re.fullmatch(r'\s*(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})(?:\s+(LAND|<LAND>))?\s*', text)
    if not match:
        raise ValueError('Expected exactly three action bins and optional LAND token')
    bins = [int(match[index]) for index in (1, 2, 3)]
    if any(value > 98 for value in bins):
        raise ValueError('AeroVLA bins must lie in 0..98')
    forward = bins[0] / 98 * 5
    down = bins[1] / 98 * 10 - 5
    yaw = bins[2] / 98 * 2.2 - 1.1
    reason = 'LAND_TOKEN' if match[4] else None
    if reason is None and forward < .01 and abs(down) < .01 and abs(yaw) < .01:
        reason = 'NEAR_ZERO_ACTION'
    return {'raw_bins': bins, 'forward_m': forward, 'down_m': down, 'yaw_rad': yaw,
            'official_termination': reason is not None, 'termination_reason': reason}


def quantization_config():
    import torch
    from transformers import BitsAndBytesConfig
    # The adapter saves full projector tensors. PEFT must wrap a float module.
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=['projector', 'lm_head'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--rgb', type=Path, required=True)
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ['HF_HOME'] = str(args.out.resolve() / 'hf-cache')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    report = {'status': 'RUNNING', 'input_mode': 'duplicate_front_diagnostic',
              'probe_program_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'original_dual_view_reproduction': False, 'autonomous_flight': False,
              'inference_completed': False, 'task_success_evaluated': False,
              'training_ready': False, 'base': str(args.base.resolve()),
              'adapter': str(args.adapter.resolve()), 'rgb': str(args.rgb.resolve()),
              'instruction': args.instruction,
              'quantization': 'NF4 double-quant, BF16 compute; projector and lm_head kept floating point'}
    result_path = args.out / 'result.json'
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor
        from peft import PeftModel

        report['rgb_sha256'] = hashlib.sha256(args.rgb.read_bytes()).hexdigest()
        report['model_input_files'] = {}
        for root in (args.base, args.adapter):
            for path in sorted(root.rglob('*')):
                if path.is_file() and '.cache' not in path.parts:
                    with path.open('rb') as source:
                        report['model_input_files'][str(path.resolve())] = hashlib.file_digest(source, 'sha256').hexdigest()
        tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=True)
        processor = AutoImageProcessor.from_pretrained(args.base, local_files_only=True, trust_remote_code=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = AutoModelForVision2Seq.from_pretrained(
            args.base, local_files_only=True, trust_remote_code=True,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={'': 0},
            attn_implementation='eager',
            quantization_config=quantization_config())
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
        model.eval()
        torch.cuda.synchronize()
        report['load_seconds'] = time.perf_counter() - started
        with Image.open(args.rgb) as source:
            front = source.convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)
        mosaic = Image.new('RGB', (224, 448))
        mosaic.paste(front, (0, 0))
        mosaic.paste(front, (0, 224))
        mosaic.save(args.out / 'input_mosaic.png')
        prompt = f'<image>\n{args.instruction}\nAction: '
        inputs = tokenizer(prompt, return_tensors='pt')
        inputs['pixel_values'] = processor(images=mosaic, return_tensors='pt')['pixel_values'].to(torch.bfloat16)
        inputs = {key: value.to('cuda:0') for key, value in inputs.items()}
        report['prompt'] = prompt
        report['input_rgb_size'] = [224, 448]
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=20, do_sample=False,
                                    eos_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        report['inference_seconds'] = time.perf_counter() - started
        report['inference_completed'] = True
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        report['gpu'] = torch.cuda.get_device_name()
        generated = output[0, inputs['input_ids'].shape[1]:]
        report['generated_token_ids'] = generated.tolist()
        report['raw_output'] = tokenizer.decode(generated, skip_special_tokens=False)
        text = report['raw_output'].replace(tokenizer.eos_token or '</s>', '').strip()
        report['action'] = parse_action(text)
        report['status'] = 'PASS'
    except Exception as error:
        report['status'] = 'FAIL'
        report['error'] = f'{type(error).__name__}: {error}'
        (args.out / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
