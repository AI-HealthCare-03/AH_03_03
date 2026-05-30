import { useEffect, useState } from "react";

import { getMySettings, updateMySettings } from "../api/settings";
import Card from "../components/Card";
import ErrorMessage from "../components/ErrorMessage";
import {
  disableBrowserPushNotifications,
  enableBrowserPushNotifications,
  getBrowserNotificationPermission,
  getBrowserPushSupport,
  getStoredFcmToken,
  subscribeForegroundMessages,
} from "../lib/firebase";

type Settings = Record<string, unknown>;

const settingLabels: Record<string, string> = {
  notification_enabled: "기본 알림",
  challenge_reminder_enabled: "챌린지 알림",
  medication_reminder_enabled: "복약/영양제 알림",
  diet_reminder_enabled: "식단 기록 알림",
};

export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings>({});
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [pushPermission, setPushPermission] = useState<NotificationPermission | "unsupported">(
    getBrowserNotificationPermission(),
  );
  const [hasFcmToken, setHasFcmToken] = useState(Boolean(getStoredFcmToken()));
  const [pushLoading, setPushLoading] = useState(false);
  const pushSupport = getBrowserPushSupport();

  const load = async () => {
    setError("");
    try {
      setSettings(await getMySettings<Settings>());
    } catch (err) {
      setError(err instanceof Error ? err.message : "설정을 불러오지 못했습니다.");
    }
  };

  useEffect(() => {
    void load();
  }, []);

  useEffect(() => {
    let unsubscribe: (() => void) | undefined;
    void subscribeForegroundMessages((payload) => {
      const title = payload.notification?.title ?? payload.data?.title ?? "브라우저 알림";
      const body = payload.notification?.body ?? payload.data?.body ?? "새 알림이 도착했습니다.";
      setNotice(`${title}: ${body}`);
    }).then((cleanup) => {
      unsubscribe = cleanup;
    });

    return () => {
      unsubscribe?.();
    };
  }, []);

  const toggle = async (key: string) => {
    const next = { [key]: !settings[key] };
    setSettings((prev) => ({ ...prev, ...next }));
    setNotice("");
    try {
      await updateMySettings(next);
      await load();
      setNotice("설정이 저장되었습니다.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "설정 저장에 실패했습니다.");
      setSettings((prev) => ({ ...prev, [key]: settings[key] }));
    }
  };

  const refreshPushState = () => {
    setPushPermission(getBrowserNotificationPermission());
    setHasFcmToken(Boolean(getStoredFcmToken()));
  };

  const enablePush = async () => {
    setPushLoading(true);
    setError("");
    setNotice("");
    try {
      await enableBrowserPushNotifications();
      refreshPushState();
      setNotice("이 브라우저에서 푸시 알림을 받을 수 있습니다.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "브라우저 알림을 허용하지 못했습니다.");
      refreshPushState();
    } finally {
      setPushLoading(false);
    }
  };

  const disablePush = async () => {
    setPushLoading(true);
    setError("");
    setNotice("");
    try {
      await disableBrowserPushNotifications();
      refreshPushState();
      setNotice("이 기기의 브라우저 알림이 해제되었습니다.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "브라우저 알림 해제에 실패했습니다.");
      refreshPushState();
    } finally {
      setPushLoading(false);
    }
  };

  const pushStateText =
    pushPermission === "unsupported"
      ? "지원 안 됨"
      : pushPermission === "granted"
        ? hasFcmToken
          ? "허용됨"
          : "권한 허용됨 · 기기 등록 필요"
        : pushPermission === "denied"
          ? "차단됨"
          : "미설정";

  return (
    <Card title="설정">
      {error && <ErrorMessage message={error} />}
      {notice && <div className="state-box">{notice}</div>}
      <div className="settings-list">
        {["notification_enabled", "challenge_reminder_enabled", "medication_reminder_enabled", "diet_reminder_enabled"].map(
          (key) => (
            <label key={key} className="toggle-row">
              <span>{settingLabels[key]}</span>
              <input checked={Boolean(settings[key])} onChange={() => void toggle(key)} type="checkbox" />
            </label>
          ),
        )}
        <div className="state-box">
          민감정보와 마케팅 수신 동의는 가입 및 계정 보안 정책에 따라 관리됩니다. 변경이 필요한 경우 1:1 문의로
          요청해주세요.
        </div>
        <div className="state-box browser-push-box">
          <div className="settings-section-header">
            <div>
              <strong>브라우저 푸시 알림</strong>
              <p className="muted">브라우저 알림을 허용하면 챌린지, 복약, 식단 기록 알림을 받을 수 있습니다.</p>
            </div>
            <span className="status-pill">{pushStateText}</span>
          </div>
          <p className="muted">알림 권한은 브라우저 설정에서 언제든 변경할 수 있습니다.</p>
          {!pushSupport.supported && <p className="muted">{pushSupport.reason}</p>}
          {pushPermission === "denied" && (
            <p className="muted">알림이 차단되어 있습니다. 브라우저 사이트 설정에서 알림 권한을 허용해주세요.</p>
          )}
          <div className="browser-push-actions">
            <button
              disabled={!pushSupport.supported || pushPermission === "denied" || pushLoading}
              onClick={() => void enablePush()}
              type="button"
            >
              {pushLoading ? "처리 중..." : "브라우저 알림 허용"}
            </button>
            <button disabled={!hasFcmToken || pushLoading} onClick={() => void disablePush()} type="button">
              이 기기 알림 해제
            </button>
          </div>
        </div>
      </div>
    </Card>
  );
}
