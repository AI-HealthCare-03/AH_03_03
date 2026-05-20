import { Navigate, Outlet } from "react-router-dom";

import { useAuth } from "./AuthContext";
import Loading from "../components/Loading";

export default function ProtectedRoute() {
  const { firebaseUser, loading } = useAuth();

  if (loading) {
    return <Loading />;
  }

  if (!firebaseUser) {
    return <Navigate to="/login" replace />;
  }

  return <Outlet />;
}
