import json

# Orden exacto de extract_features(include_hog=False, include_phog=False,
# include_context=False, include_shape=False) en src/features.py (v2).
# El set "texture" tiene 96 features (las 4 de calidad de mascara van aparte).

TEXTURE_96 = (
    ['mean', 'std', 'contrast', 'entropy']
    + [f'{r}_{m}' for r in ['SI', 'SD', 'MI', 'MD', 'II', 'ID']
       for m in ['mean', 'std', 'contrast', 'entropy']]
    + [f'glcm_{p}_{s}'
       for p in ['contrast', 'correlation', 'energy', 'homogeneity', 'dissimilarity']
       for s in ['mean', 'std']]
    + [f'lbp_{i}' for i in range(26)]
    + [f'gabor_f{freq:.1f}_t{theta}_{stat}'
       for freq in [0.1, 0.2, 0.3, 0.4]
       for theta in [0, 45, 90, 135]
       for stat in ['mean', 'std']]
)

assert len(TEXTURE_96) == 96, f"Expected 96, got {len(TEXTURE_96)}"

with open('C:\\Users\\juana\\OneDrive\\Documentos\\CAD_TBX11K\\deploy\\backend\\models\\feat_cols.json', 'w') as f:
    json.dump(TEXTURE_96, f)

print(f"feat_cols.json (v2): {len(TEXTURE_96)} features")