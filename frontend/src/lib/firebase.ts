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
const FIREBASE_MESSAGING_SW_PATH = "/firebase-messaging-sw.js";
const SERVICE_WORKER_READY_TIMEOUT_MS = 10000;

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
  const requiredEntries: Array<[keyof FirebasePublicConfig, string]> = [
    ["apiKey", "VITE_FIREBASE_API_KEY"],
    ["authDomain", "VITE_FIREBASE_AUTH_DOMAIN"],
    ["projectId", "VITE_FIREBASE_PROJECT_ID"],
    ["appId", "VITE_FIREBASE_APP_ID"],
    ["messagingSenderId", "VITE_FIREBASE_MESSAGING_SENDER_ID"],
    ["storageBucket", "VITE_FIREBASE_STORAGE_BUCKET"],
  ];
  const missingKeys = requiredEntries.filter(([key]) => !config[key]).map(([, envKey]) => envKey);
  if (!import.meta.env.VITE_FIREBASE_VAPID_KEY) {
    missingKeys.push("VITE_FIREBASE_VAPID_KEY");
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
  await assertFirebaseMessagingServiceWorkerScript();

  let registration: ServiceWorkerRegistration;
  try {
    registration = await navigator.serviceWorker.register(FIREBASE_MESSAGING_SW_PATH);
  } catch (err) {
    throw new Error("Firebase service worker 등록에 실패했습니다.");
  }

  const readyRegistration = await waitForFirebaseMessagingServiceWorkerReady();
  postFirebaseConfigToServiceWorker(readyRegistration);
  logFirebaseServiceWorkerDiagnostics(readyRegistration);
  return readyRegistration;
}

async function assertFirebaseMessagingServiceWorkerScript(): Promise<void> {
  let response: Response;
  try {
    response = await fetch(FIREBASE_MESSAGING_SW_PATH, { cache: "no-store" });
  } catch {
    throw new Error("Firebase service worker 파일에 접근하지 못했습니다.");
  }

  if (!response.ok) {
    throw new Error("Firebase service worker 파일을 불러오지 못했습니다.");
  }

  const contentType = response.headers.get("content-type") ?? "";
  const body = await response.text();
  const looksLikeHtmlFallback = contentType.includes("text/html") || /^\s*<!doctype html/i.test(body);
  if (looksLikeHtmlFallback) {
    throw new Error("Firebase service worker 경로가 index.html로 fallback되고 있습니다.");
  }
}

async function waitForFirebaseMessagingServiceWorkerReady(): Promise<ServiceWorkerRegistration> {
  try {
    return await withTimeout(
      navigator.serviceWorker.ready,
      SERVICE_WORKER_READY_TIMEOUT_MS,
      "Firebase service worker가 활성화되지 않았습니다.",
    );
  } catch (err) {
    if (err instanceof Error) {
      throw err;
    }
    throw new Error("Firebase service worker 준비 중 오류가 발생했습니다.");
  }
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, timeoutMessage: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => reject(new Error(timeoutMessage)), timeoutMs);
    promise
      .then((value) => {
        window.clearTimeout(timer);
        resolve(value);
      })
      .catch((err: unknown) => {
        window.clearTimeout(timer);
        reject(err);
      });
  });
}

function postFirebaseConfigToServiceWorker(registration: ServiceWorkerRegistration): void {
  const target = registration.active ?? registration.waiting ?? registration.installing;
  target?.postMessage({
    type: "AI_HEALTH_FIREBASE_CONFIG",
    config: getFirebasePublicConfig(),
  });
}

function logFirebaseServiceWorkerDiagnostics(registration: ServiceWorkerRegistration): void {
  if (!import.meta.env.DEV) return;
  console.info("[Firebase Web Push] service worker", {
    scriptURL: registration.active?.scriptURL ?? null,
    scope: registration.scope,
    active: Boolean(registration.active),
    installing: Boolean(registration.installing),
    waiting: Boolean(registration.waiting),
  });
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
  let token = "";
  try {
    token = await getToken(messaging, {
      vapidKey: import.meta.env.VITE_FIREBASE_VAPID_KEY,
      serviceWorkerRegistration: registration,
    });
  } catch {
    throw new Error("Firebase 브라우저 알림 토큰 발급에 실패했습니다.");
  }
  if (!token) {
    throw new Error("브라우저 알림 토큰을 발급받지 못했습니다.");
  }

  try {
    await registerFcmToken({
      token,
      platform: "web",
      device_id: getOrCreateDeviceId(),
      user_agent: navigator.userAgent,
    });
  } catch {
    throw new Error("FCM token 서버 등록에 실패했습니다.");
  }
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
