"use client";

import { motion } from "framer-motion";
import { Activity, Zap, AlertTriangle, CheckCircle } from "lucide-react";

interface StatusPanelProps {
  command: string;
  confidence: number;
  signal: string;
  fatigue: string;
  totalFwd: number;
  totalIdle: number;
  connected: boolean;
}

function StatRow({
  label,
  value,
  valueColor,
  icon,
}: {
  label: string;
  value: string;
  valueColor?: string;
  icon?: React.ReactNode;
}) {
  return (
    <div className="flex justify-between items-center py-2.5" style={{ borderBottom: "1px solid var(--border)" }}>
      <span className="text-xs uppercase tracking-widest font-medium" style={{ color: "var(--text-muted)" }}>
        {label}
      </span>
      <span
        className="text-sm font-semibold flex items-center gap-1.5"
        style={{ color: valueColor ?? "var(--text-primary)" }}
      >
        {icon}
        {value}
      </span>
    </div>
  );
}

export default function StatusPanel({
  command,
  confidence,
  signal,
  fatigue,
  totalFwd,
  totalIdle,
  connected,
}: StatusPanelProps) {
  const cmdColor =
    command === "FORWARD"
      ? "#4ADE80"
      : fatigue === "CRITICAL"
      ? "#F87171"
      : "rgba(255,255,255,0.6)";

  const fatigueColor =
    fatigue === "CRITICAL"
      ? "#F87171"
      : fatigue === "WARNING"
      ? "#FBBF24"
      : fatigue === "ALERT"
      ? "#FB923C"
      : "#4ADE80";

  const fatigueIcon =
    fatigue === "CRITICAL" || fatigue === "WARNING" || fatigue === "ALERT" ? (
      <AlertTriangle size={12} />
    ) : (
      <CheckCircle size={12} />
    );

  const total = totalFwd + totalIdle;
  const fwdPct = total > 0 ? (totalFwd / total) * 100 : 0;

  return (
    <div
      className="rounded-2xl p-5 flex flex-col gap-1"
      style={{
        background: "var(--glass-bg)",
        border: "1px solid var(--border)",
        boxShadow: "inset 0 1px 0 rgba(255,255,255,0.10)",
        backdropFilter: "blur(20px)",
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-[var(--accent)]" />
          <span className="text-[11px] font-semibold uppercase tracking-widest" style={{ color: "var(--accent)" }}>
            System Status
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <motion.div
            className="w-2 h-2 rounded-full"
            animate={{ opacity: connected ? [1, 0.3, 1] : 1 }}
            transition={{ duration: 1.5, repeat: Infinity }}
            style={{ background: connected ? "#4ADE80" : "#F87171" }}
          />
          <span className="text-[10px]" style={{ color: connected ? "#4ADE80" : "#F87171" }}>
            {connected ? "LIVE" : "OFFLINE"}
          </span>
        </div>
      </div>

      {/* Main Command */}
      <motion.div
        key={command}
        initial={{ scale: 0.92, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="flex items-center justify-center py-3 mb-2 rounded-xl"
        style={{
          background: command === "FORWARD" ? "rgba(74,222,128,0.08)" : "rgba(255,255,255,0.03)",
          border: `1px solid ${command === "FORWARD" ? "rgba(74,222,128,0.3)" : "var(--border)"}`,
        }}
      >
        <span
          className="text-2xl font-extrabold tracking-tight"
          style={{
            fontFamily: "var(--font-display)",
            color: cmdColor,
            letterSpacing: "-0.04em",
          }}
        >
          {command}
        </span>
      </motion.div>

      {/* Confidence Bar */}
      <div className="flex flex-col gap-1 mb-1">
        <div className="flex justify-between text-[10px]">
          <span style={{ color: "var(--text-muted)" }} className="uppercase tracking-widest">Confidence</span>
          <span className="font-mono" style={{ color: "var(--text-secondary)" }}>
            {(confidence * 100).toFixed(1)}%
          </span>
        </div>
        <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.06)" }}>
          <motion.div
            className="h-full rounded-full"
            animate={{ width: `${confidence * 100}%` }}
            transition={{ type: "spring", stiffness: 100, damping: 20 }}
            style={{ background: `linear-gradient(90deg, var(--accent), #4ADE80)` }}
          />
        </div>
      </div>

      {/* Stats */}
      <StatRow label="Signal" value={signal} />
      <StatRow label="Fatigue" value={fatigue} valueColor={fatigueColor} icon={fatigueIcon} />

      {/* Session Stats */}
      <div className="mt-2 pt-2">
        <div className="flex justify-between text-[10px] mb-2">
          <span style={{ color: "var(--text-muted)" }} className="uppercase tracking-widest">Session</span>
          <span className="font-mono" style={{ color: "var(--text-secondary)" }}>
            FWD {totalFwd} / IDLE {totalIdle}
          </span>
        </div>
        <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.06)" }}>
          <motion.div
            className="h-full rounded-full"
            animate={{ width: `${fwdPct}%` }}
            transition={{ type: "spring", stiffness: 60, damping: 15 }}
            style={{ background: "linear-gradient(90deg, #4ADE80aa, #4ADE80)" }}
          />
        </div>
        <div className="flex justify-between text-[9px] mt-1" style={{ color: "var(--text-muted)" }}>
          <span>IDLE</span>
          <span>FORWARD</span>
        </div>
      </div>
    </div>
  );
}
