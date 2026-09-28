"""Recompute paper tables from the supplied Excel prediction archive.

Supports ``batch*_predictions`` sheets and the supplementary per-video sheet.
It writes a compact JSON audit with counts, accuracies, MAE and adjacent/cross-
grade error totals, making the spreadsheet a reproducible source rather than a
static report.
"""
import argparse, json
from pathlib import Path
import numpy as np
from openpyxl import load_workbook

def rows_from_sheet(ws):
    values=list(ws.iter_rows(values_only=True)); header=next((r for r in values if r and 'video_id' in r),None)
    if header is None: return []
    out=[]
    for row in values[values.index(header)+1:]:
        if not row or not row[1]: continue
        d={str(k):v for k,v in zip(header[1:],row[1:]) if k}; out.append(d)
    return out

def summarize(rows, pred='pred_class_main', score='pred_grs_main'):
    label={'novice':0,'intermediate':1,'expert':2}; y=np.array([label[str(r['real_label']).lower()] for r in rows]); p=np.array([label[str(r[pred]).lower()] for r in rows]); true=np.array([float(r['real_grs']) for r in rows]); pred_score=np.array([float(r[score]) for r in rows]); err=np.abs(y-p)
    return {'n':len(rows),'correct':int((y==p).sum()),'accuracy':float((y==p).mean()),'grs_mae':float(np.abs(true-pred_score).mean()),'adjacent_errors':int((err==1).sum()),'cross_grade_errors':int((err>=2).sum())}

def main(a):
    wb=load_workbook(a.xlsx,data_only=True,read_only=True); rows=[]
    for ws in wb.worksheets:
        if ws.title.startswith('batch') and 'prediction' in ws.title: rows.extend(rows_from_sheet(ws))
    if not rows: raise SystemExit('No batch*_predictions sheets found')
    result={'sheets':wb.sheetnames,'n_rows':len(rows),'main':summarize(rows)}
    if all(k in rows[0] for k in ('pred_class_sais','pred_prob_novice_sais')): result['sais'] = summarize(rows,'pred_class_sais','pred_grs_main')
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--xlsx',required=True); p.add_argument('--output',required=True); main(p.parse_args())
