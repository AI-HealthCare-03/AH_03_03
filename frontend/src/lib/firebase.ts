import { initializeApp, type FirebaseApp } from "firebase/app";
import {
  deleteToken,
  getMessaging,
  getToken,
  isSupported,
  onMessage,
  type Messaging,
  type MessagePayload,
} from "firebase/messaging";

import { deactivateFcmToken, registerFcmToken } from "../api/notifications";

const FCM_TOKEN_STORAGE_KEY = "ai_health_fcm_token";
const FCM_DEVICE_ID_STORAGE_KEY = "ai_health_fcm_device_id";

type FirebasePublicConfig = {
  apiKey: string;
  authDomain: string;
  projectId: string;
  storageBucket: string;
  messagingSenderId: string;
  appId: string;
};

type BrowserPushSupportResult = {
  supported: boolean;
  reason?: string;
};

let firebaseApp: FirebaseApp | null = null;
let messagingPromise: Promise<Messaging | null> | null = null;

function getFirebasePublicConfig(): FirebasePublicConfig {
  return {
    apiKey: import.meta.env.VITE_FIREBASE_API_KEY ?? "",
    authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN ?? "",
    projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID ?? "",
    storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET ?? "",
    messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID ?? "",
    appId: import.meta.env.VITE_FIREBASE_APP_ID ?? "",
  };
}

function getMissingFirebaseConfigKeys(): string[] {
  const config = getFirebasePublicConfig();
  const missingKeys = Object.entries(config)
    .filter(([, value]) => !value)
    .map(([key]) => key);
  if (!import.meta.env.VITE_FIREBASE_VAPID_KEY) {
    missingKeys.push("vapidKey");
  }
  return missingKeys;
}

export function isFirebaseMessagingConfigured(): boolean {
  return getMissingFirebaseConfigKeys().length === 0;
}

export function getFirebaseMessagingConfigStatus(): { configured: boolean; missingKeys: string[] } {
  const missingKeys = getMissingFirebaseConfigKeys();
  return {
    configured: missingKeys.length === 0,
    missingKeys,
  };
}

export function getBrowserNotificationPermission(): NotificationPermission | "unsupported" {
  if (typeof window === "undefined" || !("Notification" in window)) {
    return "unsupported";
  }
  return Notification.permission;
}

export function getStoredFcmToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(FCM_TOKEN_STORAGE_KEY);
}

function setStoredFcmToken(token: string): void {
  window.localStorage.setItem(FCM_TOKEN_STORAGE_KEY, token);
}

function clearStoredFcmToken(): void {
  window.localStorage.removeItem(FCM_TOKEN_STORAGE_KEY);
}

function getOrCreateDeviceId(): string {
  const existing = window.localStorage.getItem(FCM_DEVICE_ID_STORAGE_KEY);
  if (existing) return existing;

  const generated =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `web-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  window.localStorage.setItem(FCM_DEVICE_ID_STORAGE_KEY, generated);
  return generated;
}

function clearDeviceId(): void {
  window.localStorage.removeItem(FCM_DEVICE_ID_STORAGE_KEY);
}

export function getBrowserPushSupport(): BrowserPushSupportResult {
  if (typeof window === "undefined") {
    return { supported: false, reason: "브라우저 환경에서만 푸시 알림을 사용할 수 있습니다." };
  }
  if (!("Notification" in window)) {
    return { supported: false, reason: "이 브라우저는 알림 권한을 지원하지 않습니다." };
  }
  if (!("serviceWorker" in navigator)) {
    return { supported: false, reason: "이 브라우저는 서비스 워커를 지원하지 않습니다." };
  }
  if (!("PushManager" in window)) {
    return { supported: false, reason: "이 브라우저는 브라우저 푸시를 지원하지 않습니다." };
  }
  if (!isFirebaseMessagingConfigured()) {
    return { supported: false, reason: "Firebase 웹푸시 설정이 부족합니다." };
  }
  return { supported: true };
}

function getFirebaseApp(): FirebaseApp {
  if (firebaseApp) return firebaseApp;
  firebaseApp = initializeApp(getFirebasePublicConfig());
  return firebaseApp;
}

async function getFirebaseMessaging(): Promise<Messaging | null> {
  if (!messagingPromise) {
    messagingPromise = isSupported()
      .then((supported) => (supported ? getMessaging(getFirebaseApp()) : null))
      .catch(() => null);
  }
  return messagingPromise;
}

async function registerFirebaseMessagingServiceWorker(): Promise<ServiceWorkerRegistration> {
  const config = getFirebasePublicConfig();
  const search = new URLSearchParams({
    apiKey: config.apiKey,
    authDomain: config.authDomain,
    projectId: config.projectId,
    storageBucket: config.storageBucket,
    messagingSenderId: config.messagingSenderId,
    appId: config.appId,
  });
  return navigator.serviceWorker.register(`/firebase-messaging-sw.js?${search.toString()}`);
}

export async function enableBrowserPushNotifications(): Promise<string> {
  const support = getBrowserPushSupport();
  if (!support.supported) {
    throw new Error(support.reason ?? "브라우저 푸시 알림을 사용할 수 없습니다.");
  }

  if (Notification.permission === "denied") {
    throw new Error("브라우저 설정에서 알림 권한을 허용해주세요.");
  }

  const permission = await Notification.requestPermission();
  if (permission !== "granted") {
    throw new Error("브라우저 알림 권한이 허용되지 않았습니다.");
  }

  const messaging = await getFirebaseMessaging();
  if (!messaging) {
    throw new Error("이 브라우저에서 Firebase 메시징을 사용할 수 없습니다.");
  }

  const registration = await registerFirebaseMessagingServiceWorker();
  const token = await getToken(messaging, {
    vapidKey: import.meta.env.VITE_FIREBASE_VAPID_KEY,
    serviceWorkerRegistration: registration,
  });
  if (!token) {
    throw new Error("브라우저 알림 토큰을 발급받지 못했습니다.");
  }

  await registerFcmToken({
    token,
    platform: "web",
    device_id: getOrCreateDeviceId(),
    user_agent: navigator.userAgent,
  });
  setStoredFcmToken(token);
  return token;
}

export async function disableBrowserPushNotifications(): Promise<void> {
  const storedToken = getStoredFcmToken();
  if (storedToken) {
    await deactivateFcmToken(storedToken);
  }

  const messaging = await getFirebaseMessaging();
  if (messaging) {
    await deleteToken(messaging).catch(() => undefined);
  }

  clearStoredFcmToken();
  clearDeviceId();
}

export async function subscribeForegroundMessages(
  callback: (payload: MessagePayload) => void,
): Promise<() => void> {
  const support = getBrowserPushSupport();
  if (!support.supported || getBrowserNotificationPermission() !== "granted") {
    return () => undefined;
  }

  const messaging = await getFirebaseMessaging();
  if (!messaging) {
    return () => undefined;
  }
  return onMessage(messaging, callback);
}
