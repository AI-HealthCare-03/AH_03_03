/* global firebase */
importScripts("https://www.gstatic.com/firebasejs/12.14.0/firebase-app-compat.js");
importScripts("https://www.gstatic.com/firebasejs/12.14.0/firebase-messaging-compat.js");

let firebaseMessagingInitialized = false;

function initializeFirebaseMessaging(firebaseConfig) {
  const hasFirebaseConfig = Object.values(firebaseConfig || {}).every(Boolean);
  if (!hasFirebaseConfig || !self.firebase || firebaseMessagingInitialized) {
    return;
  }

  firebase.initializeApp(firebaseConfig);
  firebaseMessagingInitialized = true;

  firebase.messaging().onBackgroundMessage((payload) => {
    const title = payload.notification?.title || payload.data?.title || "AI HealthCare 알림";
    const options = {
      body: payload.notification?.body || payload.data?.body || "새 알림이 도착했습니다.",
      icon: "/favicon.svg",
      data: payload.data || {},
    };

    self.registration.showNotification(title, options);
  });
}

self.addEventListener("message", (event) => {
  if (event.data?.type !== "AI_HEALTH_FIREBASE_CONFIG") {
    return;
  }
  initializeFirebaseMessaging(event.data.config);
});
