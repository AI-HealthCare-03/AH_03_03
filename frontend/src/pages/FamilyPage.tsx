import { FormEvent, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import {
  FamilyGroup,
  FamilyInvite,
  FamilyMember,
  FamilyMemberRole,
  FamilyRelationType,
  FamilyShareSetting,
  acceptFamilyInvite,
  acceptFamilyInviteByCode,
  addUnregisteredFamilyMember,
  createFamilyGroup,
  createFamilyInvite,
  deleteFamilyGroup,
  getFamilyGroup,
  listFamilyGroupShareSettings,
  listFamilyGroups,
  listMyFamilyInvites,
  removeFamilyMember,
  updateFamilyGroup,
  updateFamilyShareSetting,
} from "../api/family";
import { useAuth } from "../auth/AuthContext";
import Card from "../components/Card";

const relationOptions: Array<{ value: FamilyRelationType; label: string }> = [
  { value: "FATHER", label: "부" },
  { value: "MOTHER", label: "모" },
  { value: "SPOUSE", label: "배우자" },
  { value: "CHILD", label: "자녀" },
  { value: "SIBLING", label: "형제자매" },
  { value: "GRANDPARENT", label: "조부모" },
  { value: "OTHER", label: "기타" },
];

const memberRoleOptions: Array<{ value: FamilyMemberRole; label: string }> = [
  { value: "MEMBER", label: "구성원" },
  { value: "GUARDIAN", label: "보호자" },
  { value: "DEPENDENT", label: "보호 대상" },
];

const shareSettingLabels: Array<{ key: keyof FamilyShareSetting; label: string; helper: string }> = [
  { key: "share_health_records", label: "건강기록 공유", helper: "키, 체중, 혈압 등 건강기록 접근 허용" },
  { key: "share_analysis_results", label: "분석결과 공유", helper: "위험도 분석 결과 접근 허용" },
  { key: "share_diet_records", label: "식단기록 공유", helper: "식단 분석 기록 접근 허용" },
  { key: "share_medications", label: "복약정보 공유", helper: "복약/영양제 정보 접근 허용" },
  { key: "share_challenges", label: "챌린지 공유", helper: "챌린지 참여 상태 접근 허용" },
  { key: "share_exam_reports", label: "검진표 인식 결과 공유", helper: "검진표 인식 결과 접근 허용" },
  { key: "receive_analysis_alerts", label: "분석 결과 알림", helper: "가족 분석 완료 알림 수신" },
  { key: "receive_abnormal_value_alerts", label: "이상 수치 알림", helper: "혈압/혈당 등 이상 징후 알림 수신" },
  { key: "receive_medication_alerts", label: "복약 알림", helper: "복약 관련 알림 수신" },
];

const relationLabelMap: Record<string, string> = {
  SELF: "본인",
  FATHER: "부",
  MOTHER: "모",
  SPOUSE: "배우자",
  CHILD: "자녀",
  SIBLING: "형제자매",
  GRANDPARENT: "조부모",
  OTHER: "기타",
};

const roleLabelMap: Record<string, string> = {
  OWNER: "소유자",
  MEMBER: "구성원",
  GUARDIAN: "보호자",
  DEPENDENT: "보호 대상",
};

const statusLabelMap: Record<string, string> = {
  ACTIVE: "연결됨",
  INVITED: "초대 중",
  PENDING_UNREGISTERED: "미가입 가족",
  REMOVED: "해제됨",
  PENDING: "대기",
  ACCEPTED: "수락됨",
  DECLINED: "거절됨",
  EXPIRED: "만료됨",
  CANCELED: "취소됨",
};

function formatDateTime(value: string | null | undefined): string {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return new Intl.DateTimeFormat("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function normalizePhone(value: string): string {
  return value.replace(/\D/g, "");
}

export default function FamilyPage() {
  const { backendUser } = useAuth();
  const [searchParams] = useSearchParams();
  const [groups, setGroups] = useState<FamilyGroup[]>([]);
  const [selectedGroupId, setSelectedGroupId] = useState<number | null>(null);
  const [members, setMembers] = useState<FamilyMember[]>([]);
  const [invites, setInvites] = useState<FamilyInvite[]>([]);
  const [shareSettings, setShareSettings] = useState<FamilyShareSetting[]>([]);
  const [groupName, setGroupName] = useState("");
  const [renameGroupName, setRenameGroupName] = useState("");
  const [unregisteredForm, setUnregisteredForm] = useState({
    display_name: "",
    relation_type: "OTHER" as FamilyRelationType,
    phone_number: "",
    email: "",
  });
  const [inviteForm, setInviteForm] = useState({
    invitee_email: "",
    invitee_phone: "",
    relation_type: "OTHER" as FamilyRelationType,
    member_role: "MEMBER" as FamilyMemberRole,
  });
  const [inviteCodeInput, setInviteCodeInput] = useState("");
  const [latestInviteCode, setLatestInviteCode] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);

  const selectedGroup = groups.find((group) => group.id === selectedGroupId) ?? null;
  const isSelectedGroupOwner = selectedGroup?.owner_user_id === backendUser?.id;

  const summary = useMemo(() => {
    const activeMembers = members.filter((member) => member.status === "ACTIVE").length;
    const pendingInvites = invites.filter((invite) => invite.status === "PENDING").length;
    const enabledShares = shareSettings.reduce((count, setting) => {
      return count + shareSettingLabels.filter((item) => Boolean(setting[item.key])).length;
    }, 0);
    return [
      { label: "가족 그룹", value: `${groups.length}개`, helper: "내가 속한 가족 그룹입니다." },
      { label: "연결된 가족", value: `${activeMembers}명`, helper: "선택한 그룹의 연결된 가족입니다." },
      { label: "초대 대기", value: `${pendingInvites}건`, helper: "내게 도착한 대기 중 초대입니다." },
      { label: "활성 공유 권한", value: `${enabledShares}개`, helper: "명시적으로 켠 공유 권한입니다." },
    ];
  }, [groups.length, invites, members, shareSettings]);

  const loadGroups = async (preferredGroupId?: number | null) => {
    const groupItems = await listFamilyGroups();
    setGroups(groupItems);
    const nextSelectedId =
      preferredGroupId && groupItems.some((group) => group.id === preferredGroupId)
        ? preferredGroupId
        : groupItems[0]?.id ?? null;
    setSelectedGroupId(nextSelectedId);
    return nextSelectedId;
  };

  const loadSelectedGroupData = async (familyId: number | null) => {
    if (!familyId) {
      setMembers([]);
      setShareSettings([]);
      return;
    }
    const [detail, settings] = await Promise.all([getFamilyGroup(familyId), listFamilyGroupShareSettings(familyId)]);
    setMembers(detail.members);
    setShareSettings(settings);
  };

  const reloadAll = async (preferredGroupId?: number | null) => {
    setError("");
    setLoading(true);
    try {
      const nextGroupId = await loadGroups(preferredGroupId ?? selectedGroupId);
      const inviteItems = await listMyFamilyInvites();
      setInvites(inviteItems);
      await loadSelectedGroupData(nextGroupId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "가족 정보를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void reloadAll(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const inviteCode = searchParams.get("invite_code")?.trim();
    if (inviteCode) {
      setInviteCodeInput(inviteCode);
      setNotice("이메일로 받은 초대코드를 확인했습니다. 코드 수락을 눌러 가족 연결을 완료해 주세요.");
    }
  }, [searchParams]);

  const handleSelectGroup = async (familyId: number) => {
    setSelectedGroupId(familyId);
    setRenameGroupName(groups.find((group) => group.id === familyId)?.name ?? "");
    setError("");
    try {
      await loadSelectedGroupData(familyId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "가족 그룹 상세를 불러오지 못했습니다.");
    }
  };

  const runAction = async (action: () => Promise<void>, successMessage: string) => {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await action();
      setNotice(successMessage);
    } catch (err) {
      setError(err instanceof Error ? err.message : "요청을 처리하지 못했습니다.");
    } finally {
      setBusy(false);
    }
  };

  const handleCreateGroup = (event: FormEvent) => {
    event.preventDefault();
    const name = groupName.trim();
    if (!name) {
      setError("가족 그룹 이름을 입력해주세요.");
      return;
    }
    void runAction(async () => {
      const group = await createFamilyGroup({ name });
      setGroupName("");
      setRenameGroupName(group.name);
      await reloadAll(group.id);
    }, "가족 그룹을 생성했습니다.");
  };

  const handleRenameGroup = (event: FormEvent) => {
    event.preventDefault();
    if (!selectedGroup) return;
    const name = renameGroupName.trim();
    if (!name) {
      setError("변경할 가족 그룹 이름을 입력해주세요.");
      return;
    }
    void runAction(async () => {
      await updateFamilyGroup(selectedGroup.id, { name });
      await reloadAll(selectedGroup.id);
    }, "가족 그룹 이름을 변경했습니다.");
  };

  const handleDeleteGroup = () => {
    if (!selectedGroup || !window.confirm("가족 그룹을 해제하시겠습니까? 연결된 구성원과 대기 초대가 비활성화됩니다.")) {
      return;
    }
    void runAction(async () => {
      await deleteFamilyGroup(selectedGroup.id);
      setRenameGroupName("");
      await reloadAll(null);
    }, "가족 그룹을 해제했습니다.");
  };

  const handleAddUnregistered = (event: FormEvent) => {
    event.preventDefault();
    if (!selectedGroup) {
      setError("먼저 가족 그룹을 선택해주세요.");
      return;
    }
    if (!unregisteredForm.display_name.trim()) {
      setError("미가입 가족 이름을 입력해주세요.");
      return;
    }
    void runAction(async () => {
      await addUnregisteredFamilyMember(selectedGroup.id, {
        display_name: unregisteredForm.display_name.trim(),
        relation_type: unregisteredForm.relation_type,
        phone_number: normalizePhone(unregisteredForm.phone_number) || null,
        email: unregisteredForm.email.trim() || null,
      });
      setUnregisteredForm({ display_name: "", relation_type: "OTHER", phone_number: "", email: "" });
      await reloadAll(selectedGroup.id);
    }, "미가입 가족을 등록했습니다.");
  };

  const handleRemoveMember = (member: FamilyMember) => {
    if (!window.confirm(`${member.display_name}님과의 가족 연결을 해제하시겠습니까?`)) return;
    void runAction(async () => {
      await removeFamilyMember(member.id);
      await reloadAll(selectedGroupId);
    }, "가족 연결을 해제했습니다.");
  };

  const handleCreateInvite = (event: FormEvent) => {
    event.preventDefault();
    if (!selectedGroup) {
      setError("먼저 가족 그룹을 선택해주세요.");
      return;
    }
    if (!inviteForm.invitee_email.trim()) {
      setError("초대코드를 받을 이메일을 입력해주세요.");
      return;
    }
    void runAction(async () => {
      const invite = await createFamilyInvite(selectedGroup.id, {
        invitee_email: inviteForm.invitee_email.trim() || null,
        invitee_phone: normalizePhone(inviteForm.invitee_phone) || null,
        relation_type: inviteForm.relation_type,
        member_role: inviteForm.member_role,
      });
      setLatestInviteCode(invite.invite_code ?? "");
      setInviteForm({ invitee_email: "", invitee_phone: "", relation_type: "OTHER", member_role: "MEMBER" });
      await reloadAll(selectedGroup.id);
    }, "가족 초대코드를 이메일로 발송했습니다.");
  };

  const handleAcceptCode = (event: FormEvent) => {
    event.preventDefault();
    const code = inviteCodeInput.trim();
    if (!code) {
      setError("초대 코드를 입력해주세요.");
      return;
    }
    void runAction(async () => {
      await acceptFamilyInviteByCode(code);
      setInviteCodeInput("");
      await reloadAll(selectedGroupId);
    }, "가족 초대를 수락했습니다.");
  };

  const handleAcceptInvite = (invite: FamilyInvite) => {
    void runAction(async () => {
      await acceptFamilyInvite(invite.id);
      await reloadAll(selectedGroupId);
    }, "가족 초대를 수락했습니다.");
  };

  const handleToggleShareSetting = (setting: FamilyShareSetting, key: keyof FamilyShareSetting) => {
    void runAction(async () => {
      await updateFamilyShareSetting(setting.id, { [key]: !setting[key] });
      if (selectedGroupId) {
        await loadSelectedGroupData(selectedGroupId);
      }
    }, "공유 권한을 변경했습니다.");
  };

  return (
    <div className="page-stack">
      <div className="page-header">
        <div>
          <span className="badge badge-reference">Family</span>
          <h1>가족 관리</h1>
          <p>가족과 건강 정보를 안전하게 공유하고, 보호자 알림을 관리합니다.</p>
        </div>
      </div>

      <div className="state-box">
        가족 연결만으로 건강정보가 자동 공유되지 않습니다. 공유 권한을 켠 항목만 가족에게 공개됩니다.
      </div>
      {notice && <div className="state-box">{notice}</div>}
      {error && <div className="error-box">{error}</div>}

      <div className="metric-grid">
        {summary.map((item) => (
          <div className="metric-card" key={item.label}>
            <span>{item.label}</span>
            <strong>{item.value}</strong>
            <p className="muted">{item.helper}</p>
          </div>
        ))}
      </div>

      {loading ? (
        <Card>
          <p className="muted">가족 정보를 불러오는 중입니다...</p>
        </Card>
      ) : (
        <div className="family-section-grid">
          <Card title="가족 그룹">
            <form className="form compact-form" onSubmit={handleCreateGroup}>
              <label>
                새 가족 그룹 이름
                <div className="inline-form-row">
                  <input
                    className="input"
                    value={groupName}
                    onChange={(event) => setGroupName(event.target.value)}
                    placeholder="예: 우리 가족"
                  />
                  <button className="btn-primary" type="submit" disabled={busy}>
                    생성
                  </button>
                </div>
              </label>
            </form>

            {groups.length === 0 ? (
              <div className="empty-state">
                <strong>아직 가족 그룹이 없습니다.</strong>
                <p>가족 그룹을 만든 뒤 구성원 초대와 공유 권한 설정을 시작할 수 있습니다.</p>
              </div>
            ) : (
              <div className="family-group-list">
                {groups.map((group) => (
                  <button
                    className={group.id === selectedGroupId ? "family-group-item active" : "family-group-item"}
                    key={group.id}
                    type="button"
                    onClick={() => void handleSelectGroup(group.id)}
                  >
                    <strong>{group.name}</strong>
                    <span>{group.owner_user_id === backendUser?.id ? "내가 만든 그룹" : "참여 중인 그룹"}</span>
                  </button>
                ))}
              </div>
            )}

            {selectedGroup && (
              <form className="form compact-form" onSubmit={handleRenameGroup}>
                <label>
                  선택한 그룹 이름 변경
                  <div className="inline-form-row">
                    <input
                      className="input"
                      value={renameGroupName}
                      onChange={(event) => setRenameGroupName(event.target.value)}
                    />
                    <button className="btn-secondary" type="submit" disabled={busy || !isSelectedGroupOwner}>
                      저장
                    </button>
                    <button
                      className="btn-danger-outline"
                      type="button"
                      disabled={busy || !isSelectedGroupOwner}
                      onClick={handleDeleteGroup}
                    >
                      해제
                    </button>
                  </div>
                </label>
                {!isSelectedGroupOwner && <p className="muted">그룹 이름 변경과 해제는 소유자만 가능합니다.</p>}
              </form>
            )}
          </Card>

          <Card title="가족 목록">
            {!selectedGroup ? (
              <div className="empty-state">가족 그룹을 선택하면 구성원을 확인할 수 있습니다.</div>
            ) : members.length === 0 ? (
              <div className="empty-state">
                <strong>아직 연결된 가족이 없습니다.</strong>
                <p>가족을 초대하거나 미가입 가족을 직접 등록해보세요.</p>
              </div>
            ) : (
              <div className="family-member-list">
                {members.map((member) => (
                  <div className="family-member-card" key={member.id}>
                    <div>
                      <strong>{member.display_name}</strong>
                      <p className="muted">
                        {relationLabelMap[member.relation_type]} · {roleLabelMap[member.member_role]} ·{" "}
                        {statusLabelMap[member.status]}
                      </p>
                      <p className="muted">{member.email || member.phone_number || "연락처 미입력"}</p>
                    </div>
                    <div className="button-row">
                      <span className={member.is_registered ? "badge badge-saved" : "badge badge-missing"}>
                        {member.is_registered ? "가입 사용자" : "미가입 가족"}
                      </span>
                      {(isSelectedGroupOwner || member.user_id === backendUser?.id) && member.member_role !== "OWNER" && (
                        <button className="btn-danger-outline" type="button" disabled={busy} onClick={() => handleRemoveMember(member)}>
                          해제
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}

            <form className="form two-col family-form-panel" onSubmit={handleAddUnregistered}>
              <label>
                미가입 가족 이름
                <input
                  className="input"
                  value={unregisteredForm.display_name}
                  onChange={(event) => setUnregisteredForm((prev) => ({ ...prev, display_name: event.target.value }))}
                  placeholder="가족 이름"
                  disabled={!selectedGroup || !isSelectedGroupOwner}
                />
              </label>
              <label>
                관계
                <select
                  className="input"
                  value={unregisteredForm.relation_type}
                  onChange={(event) =>
                    setUnregisteredForm((prev) => ({ ...prev, relation_type: event.target.value as FamilyRelationType }))
                  }
                  disabled={!selectedGroup || !isSelectedGroupOwner}
                >
                  {relationOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                이메일
                <input
                  className="input"
                  type="email"
                  value={unregisteredForm.email}
                  onChange={(event) => setUnregisteredForm((prev) => ({ ...prev, email: event.target.value }))}
                  disabled={!selectedGroup || !isSelectedGroupOwner}
                />
              </label>
              <label>
                휴대폰 번호
                <input
                  className="input"
                  value={unregisteredForm.phone_number}
                  onChange={(event) => setUnregisteredForm((prev) => ({ ...prev, phone_number: normalizePhone(event.target.value) }))}
                  disabled={!selectedGroup || !isSelectedGroupOwner}
                />
              </label>
              <button className="btn-secondary" type="submit" disabled={busy || !selectedGroup || !isSelectedGroupOwner}>
                미가입 가족 등록
              </button>
            </form>
          </Card>

          <Card title="가족 초대">
            <div className="family-invite-grid">
              <form className="mini-card form" onSubmit={handleCreateInvite}>
                <span className="badge badge-reference">초대 생성</span>
                <label>
                  이메일(필수)
                  <input
                    className="input"
                    type="email"
                    value={inviteForm.invitee_email}
                    onChange={(event) => setInviteForm((prev) => ({ ...prev, invitee_email: event.target.value }))}
                    disabled={!selectedGroup || !isSelectedGroupOwner}
                  />
                </label>
                <label>
                  전화번호(선택)
                  <input
                    className="input"
                    value={inviteForm.invitee_phone}
                    onChange={(event) => setInviteForm((prev) => ({ ...prev, invitee_phone: normalizePhone(event.target.value) }))}
                    disabled={!selectedGroup || !isSelectedGroupOwner}
                  />
                </label>
                <label>
                  관계
                  <select
                    className="input"
                    value={inviteForm.relation_type}
                    onChange={(event) => setInviteForm((prev) => ({ ...prev, relation_type: event.target.value as FamilyRelationType }))}
                    disabled={!selectedGroup || !isSelectedGroupOwner}
                  >
                    {relationOptions.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  역할
                  <select
                    className="input"
                    value={inviteForm.member_role}
                    onChange={(event) => setInviteForm((prev) => ({ ...prev, member_role: event.target.value as FamilyMemberRole }))}
                    disabled={!selectedGroup || !isSelectedGroupOwner}
                  >
                    {memberRoleOptions.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>
                <button className="btn-primary" type="submit" disabled={busy || !selectedGroup || !isSelectedGroupOwner}>
                  이메일로 초대코드 발송
                </button>
                <p className="muted">초대코드는 입력한 이메일로 발송됩니다.</p>
              </form>

              <form className="mini-card form" onSubmit={handleAcceptCode}>
                <span className="badge badge-reference">코드 입력</span>
                <strong>초대 코드로 연결</strong>
                <p className="muted">받은 초대 코드를 입력하면 가족 그룹에 연결됩니다.</p>
                <input
                  className="input"
                  value={inviteCodeInput}
                  onChange={(event) => setInviteCodeInput(event.target.value)}
                  placeholder="초대 코드"
                />
                <button className="btn-secondary" type="submit" disabled={busy}>
                  코드 수락
                </button>
              </form>

              <div className="mini-card">
                <span className="badge badge-reference">초대 코드</span>
                {latestInviteCode ? (
                  <>
                    <strong className="invite-code-box">{latestInviteCode}</strong>
                    <p className="muted">초대 코드는 생성 직후 한 번만 표시되며, 입력한 이메일로도 발송됩니다.</p>
                  </>
                ) : (
                  <>
                    <strong>초대코드는 이메일로 발송됩니다.</strong>
                    <p className="muted">운영 환경에서는 보안을 위해 화면에 초대코드가 표시되지 않을 수 있습니다.</p>
                  </>
                )}
              </div>
            </div>

            <div className="card-list">
              {invites.length === 0 ? (
                <div className="empty-state">내게 도착한 가족 초대가 없습니다.</div>
              ) : (
                invites.map((invite) => (
                  <div className="family-member-card" key={invite.id}>
                    <div>
                      <strong>가족 초대</strong>
                      <p className="muted">
                        {relationLabelMap[invite.relation_type]} · {roleLabelMap[invite.member_role]} · 만료{" "}
                        {formatDateTime(invite.expires_at)}
                      </p>
                    </div>
                    <div className="button-row">
                      <button className="btn-primary" type="button" disabled={busy} onClick={() => handleAcceptInvite(invite)}>
                        연결
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </Card>

          <Card title="공유 권한">
            <p className="muted">공유 권한은 기본적으로 꺼져 있습니다. 켠 항목만 가족에게 공유됩니다.</p>
            {shareSettings.length === 0 ? (
              <div className="empty-state">연결된 가입 가족이 생기면 공유 권한을 설정할 수 있습니다.</div>
            ) : (
              <div className="settings-list">
                {shareSettings.map((setting) => (
                  <div className="family-share-card" key={setting.id}>
                    <div className="family-share-header">
                      <strong>
                        내 정보 #{setting.owner_user_id} → 가족 #{setting.viewer_user_id}
                      </strong>
                      {setting.owner_user_id === backendUser?.id ? (
                        <span className="badge badge-saved">변경 가능</span>
                      ) : (
                        <span className="badge badge-reference">조회 전용</span>
                      )}
                    </div>
                    {shareSettingLabels.map((item) => (
                      <label className="toggle-row family-toggle-row" key={`${setting.id}-${String(item.key)}`}>
                        <span>
                          <strong>{item.label}</strong>
                          <small>{item.helper}</small>
                        </span>
                        <input
                          type="checkbox"
                          checked={Boolean(setting[item.key])}
                          disabled={busy || setting.owner_user_id !== backendUser?.id}
                          onChange={() => handleToggleShareSetting(setting, item.key)}
                        />
                      </label>
                    ))}
                  </div>
                ))}
              </div>
            )}
          </Card>

          <Card title="가족 알림">
            <div className="empty-state">
              <strong>아직 표시할 가족 알림이 없습니다.</strong>
              <p>가족 알림은 건강분석 결과, 이상 수치, 복약 미수행, 챌린지 미수행 흐름과 함께 단계적으로 연결됩니다.</p>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
