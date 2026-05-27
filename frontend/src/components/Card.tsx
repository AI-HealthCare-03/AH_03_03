import type { ReactNode } from "react";

type CardProps = {
  title?: string;
  children: ReactNode;
  actions?: ReactNode;
};

export default function Card({ title, children, actions }: CardProps) {
  return (
    <section className="card">
      {(title || actions) && (
        <div className="card-header">
          {title && <h2>{title}</h2>}
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}
