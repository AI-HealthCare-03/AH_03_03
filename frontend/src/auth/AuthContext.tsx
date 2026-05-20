import {
  User,
  createUserWithEmailAndPassword,
  onAuthStateChanged,
  signInWithEmailAndPassword,
  signOut,
} from "firebase/auth";
import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { syncFirebaseUser, type BackendUser } from "../api/auth";
import { auth } from "../firebase";

type AuthContextValue = {
  firebaseUser: User | null;
  backendUser: BackendUser | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshBackendUser: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [firebaseUser, setFirebaseUser] = useState<User | null>(null);
  const [backendUser, setBackendUser] = useState<BackendUser | null>(null);
  const [loading, setLoading] = useState(true);

  const refreshBackendUser = async () => {
    if (!auth.currentUser) {
      setBackendUser(null);
      return;
    }
    const token = await auth.currentUser.getIdToken();
    setBackendUser(await syncFirebaseUser(token));
  };

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      setFirebaseUser(user);
      try {
        if (user) {
          const token = await user.getIdToken();
          setBackendUser(await syncFirebaseUser(token));
        } else {
          setBackendUser(null);
        }
      } finally {
        setLoading(false);
      }
    });
    return () => unsubscribe();
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      firebaseUser,
      backendUser,
      loading,
      login: async (email: string, password: string) => {
        const credential = await signInWithEmailAndPassword(auth, email, password);
        const token = await credential.user.getIdToken();
        setBackendUser(await syncFirebaseUser(token));
      },
      signup: async (email: string, password: string) => {
        const credential = await createUserWithEmailAndPassword(auth, email, password);
        const token = await credential.user.getIdToken();
        setBackendUser(await syncFirebaseUser(token));
      },
      logout: async () => {
        await signOut(auth);
        setBackendUser(null);
      },
      refreshBackendUser,
    }),
    [backendUser, firebaseUser, loading],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const value = useContext(AuthContext);
  if (!value) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return value;
}
