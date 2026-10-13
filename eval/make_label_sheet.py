"""
Build the blind label sheet for validating the judge from stored run files (no model calls).

    python eval/make_label_sheet.py eval/runs/baseline-main.jsonl eval/runs/after-slice-check-v2.jsonl

Writes eval/judge_labels.yaml (fill in `support` and `addresses_question` for each claim) and
eval/judge_labels_meta.json (stratum and dev/test split; not needed for labelling).
"""
import json
import sys
from pathlib import Path

import yaml

from earnings_rag.agent_eval import load_records
from earnings_rag.config import REPO_ROOT
from earnings_rag.judge_labels import build, select


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        raise SystemExit(__doc__)
    sheet_path = REPO_ROOT / 'eval' / 'judge_labels.yaml'
    if sheet_path.exists() and any(e.get('support') for e in yaml.safe_load(sheet_path.read_text(encoding='utf-8')) or []):
        raise SystemExit(f'{sheet_path} already has labels; move it aside to regenerate')

    records = [r for p in paths for r in load_records(p)]
    sheet, meta = build(select(records))

    header = ('# Label each claim. support: supported | partial | unsupported | not_a_claim  (is the claim backed by the\n'
              '# evidence below? numbers must match, and a company total does not support a claim about one product,\n'
              '# segment or region). addresses_question: yes | no  (does this claim help answer the question?).\n'
              '# Evidence for ids cited in the answer is shown in full (cut at 3000 characters), other retrieved\n'
              '# passages as a short preview; the judge sees everything.\n')
    sheet_path.write_text(header + yaml.safe_dump(sheet, sort_keys=False, allow_unicode=True, width=120), encoding='utf-8')
    (REPO_ROOT / 'eval' / 'judge_labels_meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(f'{len(sheet)} claims -> {sheet_path}')


if __name__ == '__main__':
    main()
