"use client";

import { useEffect, useRef } from "react";
import { motion } from "framer-motion";

interface WheelchairPosition {
  x: number;
  y: number;
}

interface ArenaProps {
  command: string;
  position: WheelchairPosition;
  fatigue: string;
}

const GRID_COLS = 24;
const GRID_ROWS = 16;

export default function Arena({ command, position, fatigue }: ArenaProps) {
  const isForward = command === "FORWARD";
  const isCritical = fatigue === "CRITICAL";

  const borderColor = isCritical
    ? "rgba(248, 113, 113, 0.5)"
    : isForward
    ? "rgba(74, 222, 128, 0.4)"
    : "rgba(255,255,255,0.08)";

  const glowColor = isCritical
    ? "rgba(248, 113, 113, 0.1)"
    : isForward
    ? "rgba(74, 222, 128, 0.08)"
    : "transparent";

  // Trail dots — last N positions shown as fading dots
  const trailRef = useRef<WheelchairPosition[]>([]);
  useEffect(() => {
    trailRef.current.push({ ...position });
    if (trailRef.current.length > 8) trailRef.current.shift();
  }, [position]);

  const trail = [...trailRef.current];

  return (
    <div
      className="relative w-full h-full rounded-3xl overflow-hidden"
      style={{
        background: `radial-gradient(ellipse at center, ${glowColor} 0%, transparent 70%), var(--bg-surface)`,
        border: `1px solid ${borderColor}`,
        boxShadow: `inset 0 1px 0 rgba(255,255,255,0.10), 0 0 40px ${glowColor}`,
        transition: "border-color 0.4s ease, box-shadow 0.4s ease",
      }}
    >
      {/* Grid */}
      <svg
        className="absolute inset-0 w-full h-full opacity-[0.04]"
        style={{ pointerEvents: "none" }}
      >
        <defs>
          <pattern
            id="grid"
            width={`${100 / GRID_COLS}%`}
            height={`${100 / GRID_ROWS}%`}
            patternUnits="userSpaceOnUse"
          >
            <path
              d={`M ${100 / GRID_COLS} 0 L 0 0 0 ${100 / GRID_ROWS}`}
              fill="none"
              stroke="white"
              strokeWidth="1"
            />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
      </svg>

      {/* Corner decorations */}
      {["top-2 left-2", "top-2 right-2", "bottom-2 left-2", "bottom-2 right-2"].map((pos, i) => (
        <div
          key={i}
          className={`absolute ${pos} w-4 h-4 opacity-30`}
          style={{
            borderTop: i < 2 ? "1px solid var(--accent)" : "none",
            borderBottom: i >= 2 ? "1px solid var(--accent)" : "none",
            borderLeft: i % 2 === 0 ? "1px solid var(--accent)" : "none",
            borderRight: i % 2 === 1 ? "1px solid var(--accent)" : "none",
          }}
        />
      ))}

      {/* Wheelchair avatar */}
      <motion.div
        className="absolute"
        animate={{
          left: `${(position.x / GRID_COLS) * 100}%`,
          top: `${(position.y / GRID_ROWS) * 100}%`,
        }}
        transition={{ type: "spring", stiffness: 200, damping: 25 }}
        style={{ translateX: "-50%", translateY: "-50%" }}
      >
        {/* Glow ring around avatar */}
        <motion.div
          className="absolute inset-0 rounded-full"
          animate={{
            scale: isForward ? [1, 1.6, 1] : 1,
            opacity: isForward ? [0.6, 0, 0.6] : 0,
          }}
          transition={{ duration: 1.2, repeat: Infinity, ease: "easeOut" }}
          style={{
            background: `radial-gradient(circle, rgba(74,222,128,0.3), transparent)`,
            width: 60,
            height: 60,
            left: "50%",
            top: "50%",
            transform: "translate(-50%,-50%)",
          }}
        />

        {/* Wheelchair SVG */}
        <motion.svg
          width="40"
          height="40"
          viewBox="0 0 24 24"
          fill="none"
          animate={{
            filter: isForward
              ? "drop-shadow(0 0 8px rgba(74,222,128,0.8))"
              : isCritical
              ? "drop-shadow(0 0 8px rgba(248,113,113,0.8))"
              : "drop-shadow(0 0 4px rgba(56,189,248,0.4))",
          }}
          transition={{ duration: 0.3 }}
        >
          {/* Body */}
          <circle cx="12" cy="4" r="2" fill={isForward ? "#4ADE80" : "#38BDF8"} />
          {/* Seat back */}
          <path d="M10 7 L10 13 L14 13 L14 7" stroke={isForward ? "#4ADE80" : "#38BDF8"} strokeWidth="1.5" fill="none" strokeLinecap="round" />
          {/* Arm + footrest */}
          <path d="M10 13 L8 16 L14 16" stroke={isForward ? "#4ADE80" : "#38BDF8"} strokeWidth="1.5" fill="none" strokeLinecap="round" />
          {/* Rear wheel */}
          <circle cx="9" cy="18" r="3" stroke={isForward ? "#4ADE80" : "#38BDF8"} strokeWidth="1.5" fill="none" />
          {/* Front wheel */}
          <circle cx="15" cy="18" r="1.5" stroke={isForward ? "#4ADE80" : "#38BDF8"} strokeWidth="1.5" fill="none" />
        </motion.svg>
      </motion.div>

      {/* Status overlay */}
      <div className="absolute bottom-3 left-0 right-0 flex justify-center">
        <motion.div
          key={command}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          className="px-4 py-1 rounded-full text-xs font-semibold tracking-widest uppercase"
          style={{
            background: isCritical
              ? "rgba(248,113,113,0.15)"
              : isForward
              ? "rgba(74,222,128,0.12)"
              : "rgba(255,255,255,0.05)",
            border: `1px solid ${isCritical ? "rgba(248,113,113,0.4)" : isForward ? "rgba(74,222,128,0.35)" : "rgba(255,255,255,0.10)"}`,
            color: isCritical ? "#F87171" : isForward ? "#4ADE80" : "rgba(255,255,255,0.5)",
          }}
        >
          {command}
        </motion.div>
      </div>
    </div>
  );
}
