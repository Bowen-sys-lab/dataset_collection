"""Predeclared metrics for a six-example diagnostic, not validation performance."""
import math


def summarize_losses(rows):
    if len(rows)!=6:raise ValueError('Exactly six diagnostic examples required')
    sums={k:[0.,0] for k in ('MOVE','HOLD','STOP')};total=0.;tokens=0
    for row in rows:
        loss=row['model_loss'];count=row['supervised_tokens'];kind=row['kind']
        if (isinstance(loss,bool) or not math.isfinite(loss) or loss<0 or type(count) is not int
            or count<=0 or kind not in sums):raise ValueError('Finite loss, supervised token count and supported kind required')
        total+=loss*count;tokens+=count;sums[kind][0]+=loss;sums[kind][1]+=1
    if any(count!=2 for _,count in sums.values()):raise ValueError('Two examples per MOVE/HOLD/STOP required')
    return {'token_weighted_loss':total/tokens,'example_mean_loss':sum(r['model_loss'] for r in rows)/6,
            'supervised_tokens':tokens,'by_kind':{k:loss/count for k,(loss,count) in sums.items()}}


def assess_fit(initial,final,*,steps):
    if type(steps) is not int or steps!=18:raise ValueError('This diagnostic requires exactly18updates')
    before=summarize_losses(initial);after=summarize_losses(final)
    if before['supervised_tokens']!=after['supervised_tokens']:raise ValueError('Comparison targets changed')
    improvements={k:before['by_kind'][k]-after['by_kind'][k] for k in before['by_kind']}
    delta=before['token_weighted_loss']-after['token_weighted_loss']
    passed=delta>=.001 and before['example_mean_loss']-after['example_mean_loss']>=.001 and min(improvements.values())>=.001
    return {'passed':passed,'criterion':'18steps; weighted, example-mean and all3kind losses improve by>=0.001',
            'initial':before,'final':after,'weighted_loss_reduction':delta,'per_kind_reduction':improvements}
