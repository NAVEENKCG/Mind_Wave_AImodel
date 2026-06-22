"use client";

import { motion } from "framer-motion";

interface BrainPower {
  theta: number;
  alpha: number;
  beta: number;
}

interface BrainPowerPanelProps {
  power: BrainPower;
}

const bands = [
  { key: "theta" as const, label: "Theta", range: "4–8 Hz", color: "#A78BFA" },
  { key: "alpha" as const, label: "Alpha", range: "8–13 Hz", color: "#38BDF8" },
  { key: "beta"  as const, label: "Beta",  range: "13–30 Hz", color: "#4ADE80" },
];

function PowerBar({ label, range, value, color }: { label: string; range: string; value: number; color: string }) {
  const pct = Math.min(100, Math.max(0, value * 100));
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex justify-between items-baseline">
        <span className="text-xs font-semibold uppercase tracking-widest" style={{ color }}>
          {label}
        </span>
        <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>
          {range}
        </span>
        <span className="text-xs font-mono" style={{ color: "var(--text-secondary)" }}>
          {pct.toFixed(0)}%
        </span>
      </div>
      <div
        className="h-2 rounded-full overflow-hidden"
        style={{ background: "rgba(255,255,255,0.06)" }}
      >
        <motion.div
          className="h-full rounded-full"
          animate={{ width: `${pct}%` }}
          transition={{ type: "spring", stiffness: 100, damping: 20 }}
          style={{
            background: `linear-gradient(90deg, ${color}99, ${color})`,
            boxShadow: `0 0 8px ${color}66`,
          }}
        />
      </div>
    </div>
  );
}

export default function BrainPowerPanel({ power }: BrainPowerPanelProps) {
  return (
    <div
      className="rounded-2xl p-5 flex flex-col gap-4"
      style={{
        background: "var(--glass-bg)",
        border: "1px solid var(--border)",
        boxShadow: "inset 0 1px 0 rgba(255,255,255,0.10)",
        backdropFilter: "blur(20px)",
      }}
    >
      <div className="flex items-center gap-2">
        <div className="w-1.5 h-1.5 rounded-full bg-[var(--accent)]" />
        <span className="text-[11px] font-semibold uppercase tracking-widest" style={{ color: "var(--accent)" }}>
          Brain Power
        </span>
      </div>
      {bands.map((b) => (
        <PowerBar
          key={b.key}
          label={b.label}
          range={b.range}
          value={power[b.key]}
          color={b.color}
        />
      ))}
    </div>
  );
}
