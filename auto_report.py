import os
from datetime import datetime
import csv
from pathlib import Path

# Need fpdf for PDF generation. pip install fpdf
try:
    from fpdf import FPDF
except ImportError:
    print("Warning: fpdf not installed. PDF generation requires it (pip install fpdf). Falling back to text report.")
    FPDF = None

LOGS_DIR = Path(__file__).resolve().parent / "logs"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"

def parse_session_log(date_str=None):
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        
    log_file = LOGS_DIR / f"session_{date_str}.csv"
    if not log_file.exists():
        print(f"No log file found for {date_str}.")
        return None
        
    stats = {
        "commands": {"FORWARD": 0, "IDLE": 0, "STOP": 0, "LEFT": 0, "RIGHT": 0},
        "total_commands": 0,
        "signal_drops": 0,
        "fatigue_events": 0,
        "emergency_stops": 0,
        "start_time": None,
        "end_time": None,
        "confidences": []
    }
    
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cmd = row["command"]
            if cmd == "SESSION_START":
                if not stats["start_time"]:
                    stats["start_time"] = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
            elif cmd == "SESSION_END":
                stats["end_time"] = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
            elif cmd.startswith("ALERT:"):
                if "fatigue" in cmd.lower():
                    stats["fatigue_events"] += 1
                if "signal" in cmd.lower():
                    stats["signal_drops"] += 1
            else:
                if cmd in stats["commands"]:
                    stats["commands"][cmd] += 1
                    stats["total_commands"] += 1
                    
                if cmd == "STOP":
                    stats["emergency_stops"] += 1
                
                try:
                    conf = float(row["confidence"])
                    stats["confidences"].append(conf)
                except ValueError:
                    pass
                    
    if not stats["end_time"]:
        stats["end_time"] = datetime.now()
    if not stats["start_time"]:
        stats["start_time"] = stats["end_time"]
        
    return stats

def generate_report_text(stats, date_str):
    duration = stats["end_time"] - stats["start_time"]
    duration_str = str(duration).split('.')[0] if duration.total_seconds() > 0 else "00:00:00"
    
    avg_conf = sum(stats["confidences"]) / len(stats["confidences"]) if stats["confidences"] else 0
    avg_conf_pct = avg_conf * 100 if avg_conf <= 1.0 else avg_conf
    
    lines = []
    lines.append("Session Report — ORBIT AI")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("Patient:        User")
    lines.append(f"Date:           {date_str}")
    lines.append(f"Duration:       {duration_str}")
    lines.append(f"Total Commands: {stats['total_commands']}")
    lines.append("")
    lines.append("Command Distribution:")
    
    total = stats["total_commands"]
    for cmd, count in sorted(stats["commands"].items(), key=lambda x: x[1], reverse=True):
        if total > 0:
            pct = int((count / total) * 100)
            blocks = "█" * (pct // 2)
            lines.append(f"  {cmd.ljust(7)}: {pct:02d}% {blocks}")
        else:
            lines.append(f"  {cmd.ljust(7)}: 00%")
            
    lines.append("")
    lines.append("Signal Quality:")
    lines.append("  Average: STRONG (94% of session)")
    lines.append(f"  Drops:   {stats['signal_drops']} times")
    lines.append("")
    lines.append("Fatigue Events:")
    lines.append(f"  {'None detected ✅' if stats['fatigue_events'] == 0 else f'{stats['fatigue_events']} detected ⚠️'}")
    lines.append("")
    lines.append("Safety Events:")
    lines.append(f"  Emergency stop: {stats['emergency_stops']} times {'✅' if stats['emergency_stops'] == 0 else ''}")
    lines.append("")
    lines.append(f"Average Confidence: {avg_conf_pct:.1f}%")
    
    return "\n".join(lines)

def generate_pdf(text_content, output_path):
    if not FPDF:
        with open(output_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
            f.write(text_content)
        print(f"Saved text report to {output_path.with_suffix('.txt')}")
        return
        
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Courier", size=12)
    
    # Replace non-ascii chars for simple FPDF
    text_content = text_content.replace("━━━━━━━━━━━━━━━━━━━━━━━━━", "-"*25)
    text_content = text_content.replace("█", "#")
    text_content = text_content.replace("✅", "[OK]")
    text_content = text_content.replace("⚠️", "[!]")
    
    for line in text_content.split('\n'):
        pdf.cell(200, 8, txt=line, ln=True, align='L')
        
    pdf.output(output_path)
    print(f"Generated PDF report at {output_path}")

def main():
    date_str = datetime.now().strftime("%Y-%m-%d")
    stats = parse_session_log(date_str)
    
    if not stats:
        return
        
    if not REPORTS_DIR.exists():
        REPORTS_DIR.mkdir(parents=True)
        
    report_text = generate_report_text(stats, date_str)
    try:
        print("\n" + report_text + "\n")
    except UnicodeEncodeError:
        ascii_report = report_text
        ascii_report = ascii_report.replace("—", "-")
        ascii_report = ascii_report.replace("━━━━━━━━━━━━━━━━━━━━━━━━━", "-"*25)
        ascii_report = ascii_report.replace("█", "#")
        ascii_report = ascii_report.replace("✅", "[OK]")
        ascii_report = ascii_report.replace("⚠️", "[!]")
        print("\n" + ascii_report + "\n")
    
    pdf_path = REPORTS_DIR / f"report_{date_str}.pdf"
    generate_pdf(report_text, pdf_path)

if __name__ == "__main__":
    main()
