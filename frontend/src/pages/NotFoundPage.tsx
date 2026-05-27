import { Link, useNavigate } from "react-router-dom";

export default function NotFoundPage() {
  const navigate = useNavigate();

  return (
    <main className="not-found-page">
      <section className="not-found-card">
        <div className="not-found-code">404</div>
        <h1>페이지를 찾을 수 없습니다.</h1>
        <p>주소가 변경되었거나 접근할 수 없는 페이지입니다. 홈으로 이동해 다시 시작해보세요.</p>
        <div className="button-row">
          <Link className="button" to="/">
            홈으로 이동
          </Link>
          <button className="button secondary" onClick={() => navigate(-1)} type="button">
            이전 페이지로 돌아가기
          </button>
        </div>
      </section>
    </main>
  );
}
