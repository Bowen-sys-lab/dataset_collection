"""Read-only per-action-token audit for the single-front Prismatic interface."""


def score_action_tokens(logits,text_labels,*,patch_tokens):
    import torch
    if (type(patch_tokens) is not int or patch_tokens<=0 or logits.ndim!=3 or text_labels.ndim!=2
        or logits.shape[0]!=1 or text_labels.shape[0]!=1
        or logits.shape[1]!=text_labels.shape[1]+patch_tokens or not torch.isfinite(logits).all()):
        raise ValueError('One finite, patch-aligned model row required')
    positions=(text_labels[0]!=-100).nonzero().flatten().tolist()
    if not positions or positions[0]<1:raise ValueError('Supervised action after BOS/prompt required')
    rows=[]
    for text_index in positions:
        # A target at text position j follows j+patch_tokens-1 causal inputs.
        index=text_index+patch_tokens-1;target=int(text_labels[0,text_index])
        if not 0<=target<logits.shape[-1]:raise ValueError('Target outside vocabulary')
        scores=logits[0,index].detach().float();logp=scores.log_softmax(-1)
        predicted=int(scores.argmax());values,indices=scores.topk(min(5,len(scores)))
        rows.append({'text_index':text_index,'logit_index':index,'target_id':target,
                     'predicted_id':predicted,'correct':predicted==target,
                     'target_rank':1+int((scores>scores[target]).sum()),
                     'target_probability':float(logp[target].exp()),
                     'predicted_probability':float(logp[predicted].exp()),
                     'nll':float(-logp[target]),
                     'top5':[{'id':int(i),'logit':float(v)} for i,v in zip(indices,values)]})
    return rows
