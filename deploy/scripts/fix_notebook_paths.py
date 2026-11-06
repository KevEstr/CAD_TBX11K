import json, glob, os, re

nb_dir = r'C:\Users\juana\OneDrive\Documentos\CAD_TBX11K\notebooks'
count = 0

for f in glob.glob(os.path.join(nb_dir, '*.ipynb')):
    nb = json.load(open(f, encoding='utf-8'))
    changed = False
    for cell in nb['cells']:
        src = ''.join(cell.get('source', []))
        new_src = re.sub(r'V2\s*/\s*"([^"]+\.png)"', r'REP / "v2" / "\1"', src)
        if new_src != src:
            cell['source'] = [new_src]
            changed = True
            count += 1
            print(f'{os.path.basename(f)}: fixed PNG ref to REP/v2')
    if changed:
        json.dump(nb, open(f, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

print(f'Fixed {count} references')