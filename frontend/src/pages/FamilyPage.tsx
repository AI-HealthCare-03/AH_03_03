import { FormEvent, useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import {
  FamilyGroup,
  FamilyInvite,
  FamilyInvitePreview,
  FamilyMember,
  FamilyMemberRole,
  FamilyRelationType,
  FamilySentInvite,
  FamilyShareSetting,
  acceptFamilyInvite,
  addUnregisteredFamilyMember,
  createFamilyGroup,
  createFamilyInvite,
  deleteFamilyGroup,
  declineFamilyInvite,
  getFamilyGroup,
  listFamilyGroupInvites,
  listFamilyGroupShareSettings,
  listFamilyGroups,
  listMyFamilyInvites,
  previewFamilyInviteByCode,
  removeFamilyMember,
  updateFamilyGroup,
  updateFamilyShareSetting,
} from "../api/family";
import { useAuth } from "../auth/AuthContext";
import Card from "../components/Card";
import ConfirmDialog from "../components/ConfirmDialog";

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
  PENDING: "대기 중",
  ACCEPTED: "수락됨",
  DECLINED: "거절됨",
  EXPIRED: "만료됨",
  CANCELED: "취소됨",
};

type InviteFeedbackDialog = {
  title: string;
  message: string;
  tone?: "default" | "danger";
  isProcessing?: boolean;
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

function familyMemberDisplayName(member: FamilyMember | undefined, fallbackUserId: number): string {
  if (!member) {
    return `가족 #${fallbackUserId}`;
  }
  const candidates = [
    member.display_name,
    (member as FamilyMember & { nickname?: string | null }).nickname,
    (member as FamilyMember & { name?: string | null }).name,
    member.email,
    member.phone_number,
  ];
  return candidates.map((value) => String(value ?? "").trim()).find(Boolean) ?? `가족 #${fallbackUserId}`;
}

function familyMemberSubLabel(member: FamilyMember | undefined): string {
  if (!member) {
    return "가족 구성원 정보를 찾을 수 없습니다.";
  }
  const relation = relationLabelMap[member.relation_type] ?? member.relation_type;
  const role = roleLabelMap[member.member_role] ?? member.member_role;
  const status = statusLabelMap[member.status] ?? member.status;
  return `${relation} · ${role} · ${status}`;
}

function getInviteFailureDialog(err: unknown): InviteFeedbackDialog {
  const detail = err instanceof Error ? err.message : "";

  if (detail.includes("본인")) {
    return {
      title: "초대할 수 없습니다.",
      message: "본인 이메일로는 가족 초대를 보낼 수 없습니다.",
      tone: "danger",
    };
  }
  if (detail.includes("이미 가족") || detail.includes("이미 연결")) {
    return {
      title: "초대할 수 없습니다.",
      message: "이미 가족으로 연결된 사용자입니다.",
      tone: "danger",
    };
  }
  if (detail.includes("이미 발송") || detail.includes("대기 중")) {
    return {
      title: "초대할 수 없습니다.",
      message: "이미 발송된 초대가 있습니다. 기존 초대를 확인해 주세요.",
      tone: "danger",
    };
  }
  if (detail.includes("이메일") || detail.includes("입력")) {
    return {
      title: "초대할 수 없습니다.",
      message: "초대할 이메일을 입력해 주세요.",
      tone: "danger",
    };
  }

  return {
    title: "초대할 수 없습니다.",
    message: "가족 초대 요청에 실패했습니다. 다시 시도해 주세요.",
    tone: "danger",
  };
}

function getInviteCodeFailureDialog(err: unknown): InviteFeedbackDialog {
  const detail = err instanceof Error ? err.message : "";

  if (detail.includes("만료")) {
    return {
      title: "초대를 확인할 수 없습니다.",
      message: "초대코드가 만료되었습니다. 초대자에게 새 코드를 요청해 주세요.",
      tone: "danger",
    };
  }
  if (detail.includes("이미 처리")) {
    return {
      title: "초대를 확인할 수 없습니다.",
      message: "이미 사용되었거나 처리된 초대입니다.",
      tone: "danger",
    };
  }
  if (detail.includes("사용할 수 없습니다") || detail.includes("발송된 초대")) {
    return {
      title: "초대를 확인할 수 없습니다.",
      message: "로그인한 계정과 초대 대상이 일치하지 않습니다.",
      tone: "danger",
    };
  }
  if (detail.includes("찾을 수 없습니다")) {
    return {
      title: "초대를 확인할 수 없습니다.",
      message: "초대코드를 찾을 수 없습니다. 가장 최근에 받은 코드를 입력해 주세요.",
      tone: "danger",
    };
  }

  return {
    title: "초대를 확인할 수 없습니다.",
    message: "가족 초대 확인에 실패했습니다. 다시 시도해 주세요.",
    tone: "danger",
  };
}

export default function FamilyPage() {
  const { backendUser } = useAuth();
  const [searchParams] = useSearchParams();
  const [groups, setGroups] = useState<FamilyGroup[]>([]);
  const [selectedGroupId, setSelectedGroupId] = useState<number | null>(null);
  const [members, setMembers] = useState<FamilyMember[]>([]);
  const [invites, setInvites] = useState<FamilyInvite[]>([]);
  const [sentInvites, setSentInvites] = useState<FamilySentInvite[]>([]);
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
  const [invitePreview, setInvitePreview] = useState<FamilyInvitePreview | null>(null);
  const [latestInviteCode, setLatestInviteCode] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [inviteFeedback, setInviteFeedback] = useState<InviteFeedbackDialog | null>(null);

  const selectedGroup = groups.find((group) => group.id === selectedGroupId) ?? null;
  const isSelectedGroupOwner = selectedGroup?.owner_user_id === backendUser?.id;
  const membersByUserId = useMemo(() => {
    const map = new Map<number, FamilyMember>();
    members.forEach((member) => {
      if (member.user_id !== null && member.user_id !== undefined) {
        map.set(Number(member.user_id), member);
      }
    });
    return map;
  }, [members]);

  const summary = useMemo(() => {
    const activeMembers = members.filter((member) => member.status === "ACTIVE").length;
    const pendingInvites = invites.filter((invite) => invite.status === "PENDING").length;
    const enabledShares = shareSettings.reduce((count, setting) => {
      return count + shareSettingLabels.filter((item) => Boolean(setting[item.key])).length;
    }, 0);
    return [
      { label: "가족 그룹", value: `${groups.length}개`, helper: "내가 속한 가족 그룹입니다." },
      { label: "일촌 수", value: `${activeMembers}명`, helper: "맺은 일촌 수입니다." },
      { label: "초대 대기", value: `${pendingInvites}건`, helper: "나에게 도착한 그룹 초대 요청입니다." },
      { label: "활성 공유 권한", value: `${enabledShares}개`, helper: "내 건강 정보를 공유 중인 항목의 개수입니다." },
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
      setSentInvites([]);
      return;
    }
    const [detail, settings, sentInviteItems] = await Promise.all([
      getFamilyGroup(familyId),
      listFamilyGroupShareSettings(familyId),
      listFamilyGroupInvites(familyId),
    ]);
    setMembers(detail.members);
    setShareSettings(settings);
    setSentInvites(sentInviteItems);
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
      setInvitePreview(null);
      setNotice("이메일로 받은 초대코드를 확인했습니다. 초대 확인을 눌러 내용을 확인해 주세요.");
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
    if (!selectedGroup || !window.confirm("가족 그룹을 해제하시겠습니까? 연결된 구성원과 대기가 중인 초대가 삭제됩니다.")) {
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
      setInviteFeedback({
        title: "초대할 수 없습니다.",
        message: "먼저 가족 그룹을 선택해 주세요.",
        tone: "danger",
      });
      return;
    }
    if (!inviteForm.invitee_email.trim()) {
      setInviteFeedback({
        title: "초대할 수 없습니다.",
        message: "초대할 이메일을 입력해 주세요.",
        tone: "danger",
      });
      return;
    }
    setBusy(true);
    setError("");
    setNotice("");
    setInviteFeedback({
      title: "초대 이메일 발송 중입니다...",
      message: "입력한 이메일로 가족 초대코드를 발송하고 있습니다.",
      isProcessing: true,
    });
    void (async () => {
      try {
        const invite = await createFamilyInvite(selectedGroup.id, {
          invitee_email: inviteForm.invitee_email.trim() || null,
          invitee_phone: normalizePhone(inviteForm.invitee_phone) || null,
          relation_type: inviteForm.relation_type,
          member_role: inviteForm.member_role,
        });
        setLatestInviteCode(invite.invite_code ?? "");
        setInviteForm({ invitee_email: "", invitee_phone: "", relation_type: "OTHER", member_role: "MEMBER" });
        await reloadAll(selectedGroup.id);
        setNotice("가족 초대 요청이 완료되었습니다.");
        setInviteFeedback({
          title: "초대 요청이 완료되었습니다.",
          message: "입력한 이메일로 가족 초대코드가 발송됩니다.",
        });
      } catch (err) {
        setInviteFeedback(getInviteFailureDialog(err));
      } finally {
        setBusy(false);
      }
    })();
  };

  const handlePreviewCode = (event: FormEvent) => {
    event.preventDefault();
    const code = inviteCodeInput.trim();
    if (!code) {
      setInviteFeedback({
        title: "초대를 확인할 수 없습니다.",
        message: "초대 코드를 입력해 주세요.",
        tone: "danger",
      });
      return;
    }
    setBusy(true);
    setError("");
    setNotice("");
    setInvitePreview(null);
    void (async () => {
      try {
        const preview = await previewFamilyInviteByCode(code);
        setInvitePreview(preview);
        setNotice("가족 초대를 확인했습니다. 내용을 확인한 뒤 수락하거나 거절해 주세요.");
      } catch (err) {
        setInviteFeedback(getInviteCodeFailureDialog(err));
      } finally {
        setBusy(false);
      }
    })();
  };

  const handleAcceptInvite = (invite: FamilyInvite) => {
    void runAction(async () => {
      await acceptFamilyInvite(invite.id);
      await reloadAll(selectedGroupId);
    }, "가족 초대를 수락했습니다.");
  };

  const handleAcceptPreviewInvite = () => {
    if (!invitePreview) return;
    void runAction(async () => {
      await acceptFamilyInvite(invitePreview.invite_id);
      setInvitePreview(null);
      setInviteCodeInput("");
      await reloadAll(invitePreview.family_id);
      setInviteFeedback({
        title: "가족 연결이 완료되었습니다.",
        message: "가족 그룹에 연결되었습니다.",
      });
    }, "가족 초대를 수락했습니다.");
  };

  const handleDeclinePreviewInvite = () => {
    if (!invitePreview) return;
    void runAction(async () => {
      await declineFamilyInvite(invitePreview.invite_id);
      setInvitePreview(null);
      setInviteCodeInput("");
      await reloadAll(selectedGroupId);
      setInviteFeedback({
        title: "가족 초대를 거절했습니다.",
        message: "초대가 거절 처리되었습니다.",
      });
    }, "가족 초대를 거절했습니다.");
  };

  const handleResendInvite = (invite: FamilySentInvite) => {
    if (!selectedGroup || !invite.invitee_email) {
      setInviteFeedback({
        title: "재발송할 수 없습니다.",
        message: "이메일이 있는 초대만 재발송할 수 있습니다.",
        tone: "danger",
      });
      return;
    }
    setInviteForm((prev) => ({
      ...prev,
      invitee_email: invite.invitee_email ?? "",
      invitee_phone: invite.invitee_phone ?? "",
      relation_type: invite.relation_type,
      member_role: invite.member_role,
    }));
    setBusy(true);
    setError("");
    setNotice("");
    setInviteFeedback({
      title: "초대 이메일 발송 중입니다...",
      message: "입력한 이메일로 가족 초대코드를 발송하고 있습니다.",
      isProcessing: true,
    });
    void (async () => {
      try {
        await createFamilyInvite(selectedGroup.id, {
          invitee_email: invite.invitee_email,
          invitee_phone: invite.invitee_phone,
          relation_type: invite.relation_type,
          member_role: invite.member_role,
        });
        await reloadAll(selectedGroup.id);
        setInviteFeedback({
          title: "초대 요청이 완료되었습니다.",
          message: "기존 초대가 새 코드로 재발송되었습니다. 가장 최근 코드만 사용할 수 있습니다.",
        });
      } catch (err) {
        setInviteFeedback(getInviteFailureDialog(err));
      } finally {
        setBusy(false);
      }
    })();
  };

  const handleToggleShareSetting = (setting: FamilyShareSetting, key: keyof FamilyShareSetting) => {
    void runAction(async () => {
      await updateFamilyShareSetting(setting.id, { [key]: !setting[key] });
      if (selectedGroupId) {
        await loadSelectedGroupData(selectedGroupId);
      }
    }, "공유 권한을 변경했습니다.");
  };

  const renderShareSettingCard = (setting: FamilyShareSetting) => {
    const ownerMember = membersByUserId.get(Number(setting.owner_user_id));
    const viewerMember = membersByUserId.get(Number(setting.viewer_user_id));
    const ownerName =
      setting.owner_user_id === backendUser?.id
        ? "내 정보"
        : familyMemberDisplayName(ownerMember, setting.owner_user_id);
    const viewerName =
      setting.viewer_user_id === backendUser?.id
        ? "나"
        : familyMemberDisplayName(viewerMember, setting.viewer_user_id);

    return (
      <div className="family-share-card" key={setting.id}>
        <div className="family-share-header">
          <div>
            <strong>
              {ownerName} → {viewerName}
            </strong>
            <p className="muted">
              {setting.viewer_user_id === backendUser?.id
                ? `${familyMemberSubLabel(ownerMember)}의 공유 설정입니다.`
                : familyMemberSubLabel(viewerMember)}
            </p>
          </div>
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
    );
  };

  return (
    <div className="page-stack">
      {inviteFeedback && (
        <ConfirmDialog
          title={inviteFeedback.title}
          message={inviteFeedback.message}
          tone={inviteFeedback.tone}
          showActions={!inviteFeedback.isProcessing}
          showCancel={false}
          onConfirm={() => setInviteFeedback(null)}
        />
      )}
      <div className="page-header">
        <div>
          <span className="badge badge-reference">Health Family</span>
          <h1>건강 일촌</h1>
          <p>소중한 사람들과 일촌을 맺고 가족 그룹을 생성해보세요. 건강 정보를 안전하게 공유하고, 보호자 알림 등 서비스를 제공합니다.</p>
          <p> 일촌 등록/가족 결성만으로 건강정보가 자동 공유되지 않습니다. 공유 권한을 켠 그룹에만 공개됩니다.</p>
        </div>
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
          <p className="muted">일촌 정보를 불러오는 중입니다...</p>
        </Card>
      ) : (
        <div className="family-section-grid">
          <Card title="가족 그룹">
            <form className="form compact-form" onSubmit={handleCreateGroup}>
              <label>
                새 가족 이름
                <div className="inline-form-row">
                  <input
                    className="input"
                    value={groupName}
                    onChange={(event) => setGroupName(event.target.value)}
                    placeholder="예: 장구네 가족"
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
              <form className="form compact-form" style={{ marginTop: 20 }} onSubmit={handleRenameGroup}>
                <label>
                  그룹 이름 변경 / 그룹 해제
                  <p className="muted" style={{ fontSize: "13px", margin: "2px 0 8px" }}>변경을 원하는 그룹을 선택해 주세요.</p>
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
            <p className="muted" style={{ fontSize: "14px", marginBottom: 12, marginTop: -4 }}>좌측에서 가족 그룹을 선택하면 구성원을 확인할 수 있습니다.</p>
            {!selectedGroup ? (
              <div className="empty-state">가족 그룹을 선택하면 구성원을 확인할 수 있습니다.</div>
            ) : members.length === 0 ? (
              <div className="empty-state">
                <strong>아직 연결된 일촌이 없습니다.</strong>
                <p>일촌을 초대해 보세요.</p>
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

          </Card>

          <div style={{ gridColumn: "1 / -1" }}>
          <Card title="가족 초대">
            <div className="family-invite-grid">
              <form className="mini-card form" onSubmit={handleCreateInvite}>
                <h3>초대 생성</h3>
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
                <p className="muted">입력한 이메일로 초대코드가 발송됩니다.</p>
              </form>

              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <form className="mini-card form" onSubmit={handlePreviewCode}>
                <h3>초대 수락</h3>
                <p className="muted">초대 코드를 입력한 뒤 그룹 정보를 확인하고 수락 여부를 선택합니다.</p>
                <input
                  className="input"
                  value={inviteCodeInput}
                  onChange={(event) => {
                    setInviteCodeInput(event.target.value);
                    setInvitePreview(null);
                  }}
                  placeholder="초대 코드(8자리 숫자)"
                />
                <button className="btn-secondary" type="submit" disabled={busy}>
                  확인
                </button>
                {invitePreview && (
                  <div className="state-box">
                    <strong>{invitePreview.inviter_display_name}님이 보낸 가족 초대입니다.</strong>
                    <p>
                      가족 그룹: {invitePreview.family_name}
                      <br />
                      초대 대상: {invitePreview.invitee_email ?? "로그인한 계정"}
                      <br />
                      만료: {formatDateTime(invitePreview.expires_at)}
                    </p>
                    <div className="button-row" style={{ marginTop: 12 }}>
                      <button className="btn-primary" type="button" disabled={busy} onClick={handleAcceptPreviewInvite}>
                        수락하기
                      </button>
                      <button className="btn-danger-outline" type="button" disabled={busy} onClick={handleDeclinePreviewInvite}>
                        거절하기
                      </button>
                    </div>
                  </div>
                )}
              </form>
              <div className="mini-card">
                <h3>보낸 초대 기록</h3>
                <p className="muted" style={{ marginBottom: 12 }}>최근 발송한 가족 초대 요청입니다.</p>
                {sentInvites.length === 0 ? (
                  <div className="empty-state">아직 보낸 가족 초대가 없습니다.</div>
                ) : (
                  <div className="card-list">
                    {sentInvites.map((invite) => (
                      <div className="family-member-card" key={invite.id}>
                        <div>
                          <strong>{invite.invitee_email || invite.invitee_phone || "초대 대상 미표시"}</strong>
                          <p className="muted">
                            {relationLabelMap[invite.relation_type]} · {roleLabelMap[invite.member_role]} ·{" "}
                            {statusLabelMap[invite.status] ?? invite.status}
                          </p>
                          <p className="muted">
                            생성 {formatDateTime(invite.created_at)} · 만료 {formatDateTime(invite.expires_at)}
                          </p>
                        </div>
                        <div className="button-row">
                          <span className="badge badge-reference">{statusLabelMap[invite.status] ?? invite.status}</span>
                          {invite.status === "PENDING" && invite.invitee_email && isSelectedGroupOwner && (
                            <button className="btn-secondary" type="button" disabled={busy} onClick={() => handleResendInvite(invite)}>
                              재발송
                            </button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="mini-card">
                <h3>받은 초대</h3>
                <p className="muted" style={{ marginBottom: 12 }}>이메일로 받은 초대코드나 대기 중인 초대를 수락할 수 있습니다.</p>
                {invites.length === 0 ? (
                  <div className="empty-state">내게 도착한 가족 초대가 없습니다.</div>
                ) : (
                  <div className="card-list">
                    {invites.map((invite) => (
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
                    ))}
                  </div>
                )}
              </div>
              </div>
            </div>
          </Card>
          </div>

          <Card title="공유 권한">
            <p className="muted">공유 권한은 기본적으로 꺼져 있습니다. 내가 직접 켠 항목만 가족에게 안전하게 공개됩니다.</p>
            {shareSettings.length === 0 ? (
              <div className="empty-state">연결된 가족이 생기면 공유 권한 설정이 가능합니다.</div>
            ) : (
              <div className="settings-list">
                {shareSettings.map(renderShareSettingCard)}
              </div>
            )}
          </Card>

          <Card title="가족 알림" actions={
            <Link className="button secondary" style={{ fontSize: "13px", padding: "4px 12px" }} to="/settings">
              알림 설정 바로가기
            </Link>
          }>
            <div className="empty-state">
              <strong>아직 표시할 가족 알림이 없습니다.</strong>
              <p>가족 알림은 내가 설정한 범위 내에서만 안전하게 공유됩니다</p>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
