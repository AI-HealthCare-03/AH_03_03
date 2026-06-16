import {
  getCanonicalRiskStage,
  getRiskStageLabel,
  riskStageOrder,
  type RiskDisplaySource,
  type RiskStageKey,
} from "../utils/riskDisplay";

export type DiseaseRiskItem = RiskDisplaySource & {
  analyzedAt?: unknown;
  analyzed_at?: unknown;
  createdAt?: unknown;
  created_at?: unknown;
  date?: unknown;
  diseaseName: string;
  id?: unknown;
};

type RiskStageBoardProps = {
  emptyMessage?: string;
  items: DiseaseRiskItem[];
  maxItemsPerStage?: number;
  showCounts?: boolean;
  showEmptyStages?: boolean;
  title?: string;
  variant?: "default" | "compact" | "preview" | "grid2x2";
};

function stageClassName(stage: RiskStageKey): string {
  return `risk-stage-card risk-stage-${stage.toLowerCase().replaceAll("_", "-")}`;
}

function getItemTimestamp(item: DiseaseRiskItem): number {
  const candidates = [item.analyzedAt, item.analyzed_at, item.createdAt, item.created_at, item.date];
  for (const candidate of candidates) {
    const parsed = Date.parse(String(candidate ?? ""));
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return 0;
}

function getItemId(item: DiseaseRiskItem): number {
  const parsed = Number(item.id);
  return Number.isFinite(parsed) ? parsed : 0;
}

function isNewerItem(candidate: DiseaseRiskItem, candidateIndex: number, current: DiseaseRiskItem, currentIndex: number): boolean {
  const candidateTimestamp = getItemTimestamp(candidate);
  const currentTimestamp = getItemTimestamp(current);
  if (candidateTimestamp !== currentTimestamp) {
    return candidateTimestamp > currentTimestamp;
  }
  const candidateId = getItemId(candidate);
  const currentId = getItemId(current);
  if (candidateId !== currentId) {
    return candidateId > currentId;
  }
  return candidateIndex > currentIndex;
}

function dedupeLatestByDisease(items: DiseaseRiskItem[]): DiseaseRiskItem[] {
  const latestByDisease = new Map<string, { index: number; item: DiseaseRiskItem }>();
  items.forEach((item, index) => {
    const diseaseName = item.diseaseName.trim();
    if (!diseaseName) {
      return;
    }
    const key = diseaseName.toLocaleLowerCase("ko-KR");
    const normalizedItem = { ...item, diseaseName };
    const current = latestByDisease.get(key);
    if (!current || isNewerItem(normalizedItem, index, current.item, current.index)) {
      latestByDisease.set(key, { index, item: normalizedItem });
    }
  });
  return Array.from(latestByDisease.values()).map(({ item }) => item);
}

export default function RiskStageBoard({
  emptyMessage = "표시할 질환 결과가 없습니다.",
  items,
  maxItemsPerStage,
  showCounts,
  showEmptyStages,
  title,
  variant = "default",
}: RiskStageBoardProps) {
  const shouldShowCounts = showCounts ?? (variant === "default" || variant === "grid2x2");
  const shouldShowEmptyStages = showEmptyStages ?? (variant === "default" || variant === "grid2x2");
  const latestDiseaseItems = dedupeLatestByDisease(items);
  const grouped = riskStageOrder.reduce<Record<RiskStageKey, DiseaseRiskItem[]>>(
    (acc, stage) => ({ ...acc, [stage]: [] }),
    {
      LOW: [],
      ATTENTION: [],
      CAUTION: [],
      HIGH_CAUTION: [],
    },
  );

  latestDiseaseItems.forEach((item) => {
    grouped[getCanonicalRiskStage(item)].push(item);
  });

  if (latestDiseaseItems.length === 0) {
    return <div className="risk-stage-board-empty">{emptyMessage}</div>;
  }

  const visibleStages = riskStageOrder.filter((stage) => shouldShowEmptyStages || grouped[stage].length > 0);

  return (
    <div className={`risk-stage-board risk-stage-board-${variant}`} aria-label={title ?? "질환별 관리 필요 단계"}>
      {title ? <p className="risk-stage-board-title">{title}</p> : null}
      {visibleStages.map((stage) => {
        const stageItems = maxItemsPerStage ? grouped[stage].slice(0, maxItemsPerStage) : grouped[stage];
        const hiddenCount = maxItemsPerStage ? Math.max(grouped[stage].length - maxItemsPerStage, 0) : 0;
        return (
          <section className={stageClassName(stage)} key={stage}>
            <div className="risk-stage-header">
              <span>{getRiskStageLabel(stage)}</span>
              {shouldShowCounts ? (
                <em className={grouped[stage].length === 0 ? "is-empty" : undefined}>{grouped[stage].length}개</em>
              ) : null}
            </div>
            <div className="risk-stage-panel">
              <div className="risk-stage-chip-list">
                {stageItems.length > 0 ? (
                  <>
                    {stageItems.map((item) => (
                      <span className="risk-stage-chip" key={`${stage}-${item.diseaseName}`} title={item.diseaseName}>
                        {item.diseaseName}
                      </span>
                    ))}
                    {hiddenCount > 0 ? <span className="risk-stage-more">+{hiddenCount}</span> : null}
                  </>
                ) : (
                  <span className="risk-stage-empty">해당 없음</span>
                )}
              </div>
            </div>
          </section>
        );
      })}
    </div>
  );
}
