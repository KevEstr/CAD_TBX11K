import json

# Orden exacto de extract_features() en src/features.py:
# feat_intensity_global -> feat_intensity_regional -> feat_glcm -> feat_lbp -> feat_hog -> feat_gabor -> feat_phog

feat_cols = (
    # 1. Intensidad global (4)
    ['mean', 'std', 'contrast', 'entropy']

    # 2. Intensidad regional (24)
    + [f'{r}_{m}' for r in ['SI', 'SD', 'MI', 'MD', 'II', 'ID']
       for m in ['mean', 'std', 'contrast', 'entropy']]

    # 3. GLCM (10)
    + [f'glcm_{p}_{s}'
       for p in ['contrast', 'correlation', 'energy', 'homogeneity', 'dissimilarity']
       for s in ['mean', 'std']]

    # 4. LBP (26)
    + [f'lbp_{i}' for i in range(26)]

    # 5. HOG (1764)
    + [f'hog_{i}' for i in range(1764)]

    # 6. Gabor (32)
    + [f'gabor_f{freq:.1f}_t{theta}_{stat}'
       for freq in [0.1, 0.2, 0.3, 0.4]
       for theta in [0, 45, 90, 135]
       for stat in ['mean', 'std']]

    # 7. PHOG (1440)
    + [f'phog_l0_{i}' for i in range(288)]
    + [f'phog_l1_{i}' for i in range(1152)]
)

assert len(feat_cols) == 3300, f"Expected 3300 features, got {len(feat_cols)}"

with open('C:\\Users\\juana\\OneDrive\\Documentos\\CAD_TBX11K\\deploy\\backend\\models\\feat_cols.json', 'w') as f:
    json.dump(feat_cols, f)

print(f"feat_cols.json generated: {len(feat_cols)} features")