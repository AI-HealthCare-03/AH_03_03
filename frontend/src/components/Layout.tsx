import { Outlet } from "react-router-dom";

import Navbar from "./Navbar";
import Sidebar from "./Sidebar";

export default function Layout() {
  return (
    <>
      <Navbar />
      <div className="shell">
        <Sidebar />
        <main className="content">
          <Outlet />
        </main>
      </div>
    </>
  );
}
