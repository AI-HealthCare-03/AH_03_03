"""
KNHANES 2024 건강상태 클러스터링 파이프라인
방식 C: 임상 수치 + 진단 코드 동시 투입
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# ── 출력 폴더 설정 (코드 파일과 같은 폴더에 outputs/ 생성) ──
OUTPUT_DIR = Path(__file__).parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 0. 한글 폰트 ──────────────────────────────────────────
import platform
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'Apple SD Gothic Neo'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:  # Linux
    fm._load_fontmanager(try_read_cache=False)
    nanum = [f for f in fm.findSystemFonts() if 'Nanum' in f]
    if nanum:
        plt.rcParams['font.family'] = fm.FontProperties(fname=nanum[0]).get_name()
plt.rcParams['axes.unicode_minus'] = False

# ── 1. 데이터 로드 ─────────────────────────────────────────
print("▶ 데이터 로드 중...")
# ★ 데이터 경로 여기만 바꾸세요 ★
DATA_PATH = '/Users/admin/PycharmProjects/AH_03_03/ai_worker/data/hn24_all.sas7bdat'
df = pd.read_sas(DATA_PATH)
print(f"  원본 shape: {df.shape}")

# ── 2. 변수 정의 ───────────────────────────────────────────
clinical = [
    'HE_sbp', 'HE_dbp',
    'HE_glu', 'HE_HbA1c',
    'HE_chol', 'HE_TG', 'HE_HDL_st2', 'HE_LDL_drct',
    'HE_BMI', 'HE_wc',
    'HE_ast', 'HE_alt',
    'HE_crea', 'HE_BUN', 'HE_Uacid',
    'HE_HB', 'HE_WBC', 'HE_RBC', 'HE_Bplt',
    'HE_Upro',
]

diag_raw = ['DI1_dg', 'DI2_dg', 'DM4_dg', 'DI3_dg', 'DI4_dg', 'DI5_dg', 'DE1_dg', 'DN1_dg']
diag_label = {
    'DI1_dg': '고혈압', 'DI2_dg': '이상지질혈증', 'DM4_dg': '당뇨',
    'DI3_dg': '뇌졸중', 'DI4_dg': '심근경색', 'DI5_dg': '협심증',
    'DE1_dg': '간질환', 'DN1_dg': '신장질환'
}

# 해석용 변수 (클러스터링 투입 X)
profile_vars = ['sex', 'age', 'sm_presnt', 'BD1', 'pa_aerobic']

# ── 3. 진단코드 전처리: 8/9 → NaN, 1=진단, 0=없음 ──────────
print("▶ 진단코드 전처리 중...")
df_diag = df[diag_raw].copy()
for col in diag_raw:
    df_diag[col] = df_diag[col].replace({8.0: np.nan, 9.0: np.nan})
    df_diag[col] = df_diag[col].where(df_diag[col].isin([0.0, 1.0]), np.nan)

# ── 4. 분석 데이터셋 구성 ──────────────────────────────────
all_features = clinical + diag_raw
df_feat = pd.concat([df[clinical], df_diag], axis=1)

# 임상수치 결측 50% 초과 행 제거
df_feat['clinical_missing'] = df[clinical].isnull().mean(axis=1)
df_feat = df_feat[df_feat['clinical_missing'] <= 0.5].drop(columns='clinical_missing')
print(f"  결측 50% 초과 행 제거 후: {df_feat.shape[0]}명")

# ── 4-1. 임상 범위 이상치 → NaN 처리 ─────────────────────
print("▶ 임상 범위 이상치 NaN 처리 중...")
CLINICAL_BOUNDS = {
    'HE_sbp':      (60,  250),
    'HE_dbp':      (30,  150),
    'HE_glu':      (50,  500),
    'HE_HbA1c':    (3.5,  15),
    'HE_chol':     (50,  500),
    'HE_TG':       (10,  500),
    'HE_HDL_st2':  (10,  150),
    'HE_LDL_drct': (10,  400),
    'HE_BMI':      (13,   60),
    'HE_wc':       (40,  160),
    'HE_ast':      (5,   200),
    'HE_alt':      (5,   200),
    'HE_crea':     (0.3,   5),
    'HE_BUN':      (2,    80),
    'HE_Uacid':    (1,    15),
    'HE_HB':       (5,    22),
    'HE_WBC':      (1,    30),
    'HE_RBC':      (2,     8),
    'HE_Bplt':     (50,  700),
    'HE_Upro':     (0,     4),
}

outlier_counts = {}
for col, (lo, hi) in CLINICAL_BOUNDS.items():
    if col in df_feat.columns:
        mask = (df_feat[col] < lo) | (df_feat[col] > hi)
        outlier_counts[col] = mask.sum()
        df_feat.loc[mask, col] = np.nan

total_outliers = sum(outlier_counts.values())
print(f"  이상치 → NaN 처리: 총 {total_outliers}개 값")
for col, cnt in outlier_counts.items():
    if cnt > 0:
        print(f"    {col}: {cnt}개")

# ── 5. KNN Imputation ─────────────────────────────────────
from sklearn.impute import KNNImputer
print("▶ KNN Imputation 중 (n_neighbors=5, 시간 소요)...")

imp_knn = KNNImputer(n_neighbors=5)
arr_clinical = imp_knn.fit_transform(df_feat[clinical])
print(f"  임상수치 KNN imputation 완료")

# 진단코드: 0으로 채움 (진단 정보 없음 = 없는 것으로)
arr_diag = df_feat[diag_raw].fillna(0).values

# 진단코드는 클러스터링 투입 제외 → 사후 해석용으로만 사용
arr_all = arr_clinical  # 임상수치만 투입
print(f"  최종 피처 행렬 (임상수치만): {arr_all.shape}")

# ── 6. 스케일링 ────────────────────────────────────────────
print("▶ RobustScaler 적용 중...")
scaler = RobustScaler()
arr_scaled = scaler.fit_transform(arr_all)

# ── 7. 최적 K 탐색 (Elbow + Silhouette) ───────────────────
print("▶ 최적 K 탐색 중 (K=2~8)...")
k_range = range(2, 9)
inertias, sil_scores, db_scores = [], [], []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(arr_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(arr_scaled, labels, sample_size=3000, random_state=42))
    db_scores.append(davies_bouldin_score(arr_scaled, labels))
    print(f"  K={k}: inertia={km.inertia_:.0f}, silhouette={sil_scores[-1]:.4f}, DB={db_scores[-1]:.4f}")

# ── 8. Elbow + Silhouette 시각화 ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('최적 클러스터 수 탐색 (임상수치만)', fontsize=13)

axes[0].plot(list(k_range), inertias, 'bo-')
axes[0].set_title('Elbow Method (Inertia)')
axes[0].set_xlabel('K'); axes[0].set_ylabel('Inertia')
axes[0].grid(alpha=0.3)

axes[1].plot(list(k_range), sil_scores, 'go-')
axes[1].set_title('Silhouette Score (높을수록 좋음)')
axes[1].set_xlabel('K'); axes[1].set_ylabel('Silhouette')
axes[1].grid(alpha=0.3)

axes[2].plot(list(k_range), db_scores, 'ro-')
axes[2].set_title('Davies-Bouldin Score (낮을수록 좋음)')
axes[2].set_xlabel('K'); axes[2].set_ylabel('DB Score')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / '01_optimal_k.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → 01_optimal_k.png 저장")

# ── 9. 최적 K로 최종 클러스터링 ───────────────────────────
best_k = 6  # K 고정
print(f"\n▶ K={best_k} 고정으로 최종 클러스터링...")
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
labels_final = km_final.fit_predict(arr_scaled)

df_result = df_feat.copy().reset_index(drop=True)
df_result['cluster'] = labels_final

# 해석용 변수 붙이기
df_result = pd.concat([df_result, df[profile_vars].reset_index(drop=True)], axis=1)

# ── 10. 클러스터별 임상수치 프로파일 ──────────────────────
print("▶ 클러스터 프로파일 생성 중...")
profile = df_result.groupby('cluster')[clinical].median()
print("\n=== 클러스터별 임상수치 중앙값 ===")
print(profile.round(1).to_string())

# ── 11. 진단코드 분포 ─────────────────────────────────────
print("\n=== 클러스터별 진단율 (%) ===")
diag_rate = df_result.groupby('cluster')[diag_raw].apply(lambda x: (x == 1).mean() * 100)
diag_rate.columns = [diag_label[c] for c in diag_raw]
print(diag_rate.round(1).to_string())

# ── 12. 히트맵 시각화 ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle(f'클러스터 프로파일 (K={best_k}, 임상수치만)', fontsize=13)

# 임상수치 히트맵 (Z-score 기준)
profile_z = (profile - profile.mean()) / profile.std()
sns.heatmap(profile_z.T, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, ax=axes[0], cbar_kws={'label': 'Z-score'})
axes[0].set_title('임상수치 Z-score (중앙값 기준)')
axes[0].set_xlabel('클러스터')

# 진단율 히트맵
sns.heatmap(diag_rate.T, annot=True, fmt='.1f', cmap='YlOrRd',
            ax=axes[1], cbar_kws={'label': '진단율 (%)'})
axes[1].set_title('클러스터별 질병 진단율 (%)')
axes[1].set_xlabel('클러스터')

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / '02_cluster_profile.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → 02_cluster_profile.png 저장")

# ── 13. PCA 2D 시각화 ─────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
arr_pca = pca.fit_transform(arr_scaled)
explained = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(9, 7))
colors = plt.cm.tab10(np.linspace(0, 1, best_k))
for i in range(best_k):
    mask = labels_final == i
    ax.scatter(arr_pca[mask, 0], arr_pca[mask, 1],
               c=[colors[i]], label=f'Cluster {i} (n={mask.sum()})',
               alpha=0.4, s=10)
ax.set_title(f'PCA 2D 시각화 (설명분산: {explained[0]:.1%} + {explained[1]:.1%} = {sum(explained):.1%})')
ax.set_xlabel(f'PC1 ({explained[0]:.1%})')
ax.set_ylabel(f'PC2 ({explained[1]:.1%})')
ax.legend(markerscale=3)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / '03_pca_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → 03_pca_scatter.png 저장")

# ── 14. 클러스터 크기 요약 ────────────────────────────────
print("\n=== 클러스터 크기 ===")
print(df_result['cluster'].value_counts().sort_index())
print("\n✅ 완료!")
