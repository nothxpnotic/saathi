"""
====================================================================
SAATHI — Smart AI-Assisted Triage and Healthcare Interface
====================================================================
Supporting Evidence: Prototype Desktop / Tablet UI
Author  : Muhammad Aayan Malik et al.
Org     : Nishtar Medical University, Multan, Pakistan

DESCRIPTION
-----------
This file demonstrates what the SAATHI user interface looks like
as a Python prototype (Tkinter).  The final product targets Android
tablets; this desktop prototype is submitted as supporting evidence
of the planned UI/UX flow and feature set.

It demonstrates:
  • Multilingual label toggling (English ↔ Urdu transliteration)
  • Vital-sign entry with real-time range validation
  • Chief-complaint selection via dropdown
  • Gradient-Boosted Classifier inference (loads saved model)
  • Colour-coded triage result card  (GREEN / AMBER / RED)
  • Clinical flag display
  • Print-to-slip simulation (writes a plain-text OPD slip)

RUN
---
    python saathi_ui.py
    (Requires saathi_model.py to have been run first to generate
     models/saathi_gbc_model.pkl)
====================================================================
"""

import os
import sys
import datetime
import tkinter as tk
from tkinter import ttk, messagebox

# ── Try to import inference helper; fall back to rule-based demo ──
try:
    from saathi_model import predict_triage, COMPLAINTS
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    COMPLAINTS = {
        "chest_pain":          0,
        "shortness_of_breath": 1,
        "high_fever":          2,
        "abdominal_pain":      3,
        "head_injury":         4,
        "altered_sensorium":   5,
        "vomiting_diarrhoea":  6,
        "laceration_wound":    7,
        "general_weakness":    8,
        "routine_followup":    9,
    }

# ─────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────
COLOURS = {
    "bg":          "#F4F7FB",
    "panel":       "#FFFFFF",
    "header_bg":   "#0D3B66",
    "header_fg":   "#FFFFFF",
    "accent":      "#1A6B9A",
    "border":      "#D0DBE8",
    "Immediate":   "#C0392B",
    "Urgent":      "#E67E22",
    "Routine":     "#27AE60",
    "label_fg":    "#2C3E50",
    "entry_bg":    "#EBF2FA",
    "entry_focus": "#1A6B9A",
    "btn_bg":      "#0D3B66",
    "btn_fg":      "#FFFFFF",
    "btn_hover":   "#1A6B9A",
}

FONTS = {
    "title":   ("Helvetica", 16, "bold"),
    "subtitle":("Helvetica", 10),
    "section": ("Helvetica", 11, "bold"),
    "label":   ("Helvetica", 10),
    "entry":   ("Helvetica", 11),
    "result":  ("Helvetica", 22, "bold"),
    "small":   ("Helvetica", 9),
    "btn":     ("Helvetica", 11, "bold"),
    "flag":    ("Helvetica", 9, "italic"),
}

# ─────────────────────────────────────────────────────────────────
# TRANSLATIONS  (English ↔ Urdu transliteration)
# ─────────────────────────────────────────────────────────────────
LANG = {
    "en": {
        "title":        "SAATHI — OPD Triage System",
        "subtitle":     "Smart AI-Assisted Triage & Healthcare Interface",
        "patient":      "Patient Details",
        "name":         "Patient Name",
        "age":          "Age (years)",
        "gender":       "Gender",
        "vitals":       "Vital Signs",
        "spo2":         "SpO₂ (%)",
        "sbp":          "Systolic BP (mmHg)",
        "dbp":          "Diastolic BP (mmHg)",
        "temp":         "Temperature (°C)",
        "hr":           "Heart Rate (bpm)",
        "complaint":    "Chief Complaint",
        "triage_btn":   "▶  RUN TRIAGE",
        "clear_btn":    "✕  CLEAR",
        "print_btn":    "🖨  PRINT SLIP",
        "result_lbl":   "Triage Result",
        "confidence":   "Confidence",
        "flags":        "Clinical Flags",
        "male":         "Male",
        "female":       "Female",
        "toggle":       "🌐  اردو",
        "complaints": {
            "chest_pain":          "Chest Pain",
            "shortness_of_breath": "Shortness of Breath",
            "high_fever":          "High Fever",
            "abdominal_pain":      "Abdominal Pain",
            "head_injury":         "Head Injury",
            "altered_sensorium":   "Altered Sensorium",
            "vomiting_diarrhoea":  "Vomiting / Diarrhoea",
            "laceration_wound":    "Laceration / Wound",
            "general_weakness":    "General Weakness",
            "routine_followup":    "Routine Follow-up",
        },
    },
    "ur": {
        "title":        "ساتھی — OPD ٹریاج سسٹم",
        "subtitle":     "اسمارٹ اے آئی معاون ٹریاج اور صحت انٹرفیس",
        "patient":      "مریض کی تفصیل",
        "name":         "مریض کا نام",
        "age":          "عمر (سال)",
        "gender":       "جنس",
        "vitals":       "علامات حیاتی",
        "spo2":         "آکسیجن سیچوریشن (%)",
        "sbp":          "سسٹولک بلڈ پریشر (mmHg)",
        "dbp":          "ڈائیسٹولک بلڈ پریشر (mmHg)",
        "temp":         "درجہ حرارت (°C)",
        "hr":           "دل کی دھڑکن (bpm)",
        "complaint":    "اہم شکایت",
        "triage_btn":   "▶  ٹریاج چلائیں",
        "clear_btn":    "✕  صاف کریں",
        "print_btn":    "🖨  پرچی پرنٹ کریں",
        "result_lbl":   "ٹریاج نتیجہ",
        "confidence":   "اطمینان",
        "flags":        "طبی انتباہات",
        "male":         "مرد",
        "female":       "عورت",
        "toggle":       "🌐  English",
        "complaints": {
            "chest_pain":          "سینے میں درد",
            "shortness_of_breath": "سانس کی تنگی",
            "high_fever":          "تیز بخار",
            "abdominal_pain":      "پیٹ میں درد",
            "head_injury":         "سر کی چوٹ",
            "altered_sensorium":   "بے ہوشی / ذہنی الجھن",
            "vomiting_diarrhoea":  "الٹی / اسہال",
            "laceration_wound":    "زخم / کٹاؤ",
            "general_weakness":    "عمومی کمزوری",
            "routine_followup":    "معمول کا فالو اپ",
        },
    },
}


# ─────────────────────────────────────────────────────────────────
# FALLBACK RULE-BASED TRIAGE  (when model is not trained yet)
# ─────────────────────────────────────────────────────────────────

def rule_based_triage(spo2, sbp, dbp, temp, hr, complaint_code):
    flags = []
    if spo2 < 90:       flags.append("⚠ Critical SpO₂ — consider oxygen therapy")
    if sbp < 80:        flags.append("⚠ Hypotensive — urgent BP management")
    if sbp > 180:       flags.append("⚠ Hypertensive crisis — immediate review")
    if hr > 150:        flags.append("⚠ Tachycardia — arrhythmia screen")
    if hr < 40:         flags.append("⚠ Bradycardia — cardiac monitoring")
    if temp > 40.5:     flags.append("⚠ Hyperpyrexia — antipyretics + workup")

    complaint_name = [k for k, v in COMPLAINTS.items() if v == complaint_code]
    complaint_name = complaint_name[0] if complaint_name else ""

    if (spo2 < 90 or sbp < 80 or sbp > 180 or hr > 150 or hr < 40
            or temp > 40.5
            or complaint_name in ("chest_pain", "altered_sensorium", "head_injury")):
        label, conf = "Immediate", 0.91
    elif (90 <= spo2 < 94 or 80 <= sbp <= 90 or 160 <= sbp <= 180
            or 110 <= hr <= 150 or 40 <= hr <= 50 or 38.5 <= temp <= 40.5
            or complaint_name in ("shortness_of_breath", "high_fever", "abdominal_pain")):
        label, conf = "Urgent", 0.85
    else:
        label, conf = "Routine", 0.93

    return {
        "label": label,
        "confidence": conf,
        "probabilities": {},
        "flags": flags if flags else ["No critical flags detected"],
    }


# ─────────────────────────────────────────────────────────────────
# MAIN APPLICATION CLASS
# ─────────────────────────────────────────────────────────────────

class SAATHIApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.lang_code  = "en"
        self.last_result = None

        self.title("SAATHI — OPD Triage System")
        self.geometry("900x680")
        self.minsize(820, 620)
        self.configure(bg=COLOURS["bg"])
        self.resizable(True, True)

        self._build_ui()
        self._apply_language()

    # ── UI CONSTRUCTION ──────────────────────────────────────────

    def _build_ui(self):
        # ── Header bar ────────────────────────────────────────────
        hdr = tk.Frame(self, bg=COLOURS["header_bg"], height=72)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        self.lbl_title    = tk.Label(hdr, text="", bg=COLOURS["header_bg"],
                                     fg=COLOURS["header_fg"], font=FONTS["title"])
        self.lbl_subtitle = tk.Label(hdr, text="", bg=COLOURS["header_bg"],
                                     fg="#A8C4DC", font=FONTS["subtitle"])
        self.lbl_title.pack(anchor="w", padx=20, pady=(12, 0))
        self.lbl_subtitle.pack(anchor="w", padx=20)

        self.btn_lang = tk.Button(hdr, text="", font=FONTS["small"],
                                  bg=COLOURS["accent"], fg="white",
                                  relief="flat", cursor="hand2",
                                  command=self._toggle_language)
        self.btn_lang.place(relx=1.0, rely=0.5, anchor="e", x=-18)

        # ── Main content (left panel + right result panel) ────────
        content = tk.Frame(self, bg=COLOURS["bg"])
        content.pack(fill="both", expand=True, padx=16, pady=12)

        left  = tk.Frame(content, bg=COLOURS["bg"])
        right = tk.Frame(content, bg=COLOURS["bg"], width=260)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self._build_left(left)
        self._build_right(right)

    def _card(self, parent, pady=(0, 10)):
        """White rounded-looking card."""
        f = tk.Frame(parent, bg=COLOURS["panel"],
                     highlightbackground=COLOURS["border"],
                     highlightthickness=1)
        f.pack(fill="x", pady=pady)
        return f

    def _section_label(self, parent, text_key):
        lbl = tk.Label(parent, text="", bg=COLOURS["panel"],
                       fg=COLOURS["accent"], font=FONTS["section"])
        lbl.pack(anchor="w", padx=14, pady=(10, 4))
        lbl._text_key = text_key
        self._lang_widgets.append(lbl)
        return lbl

    def _field_row(self, parent, label_key, widget):
        """Two-column row: label on left, widget on right."""
        row = tk.Frame(parent, bg=COLOURS["panel"])
        row.pack(fill="x", padx=14, pady=4)
        lbl = tk.Label(row, text="", bg=COLOURS["panel"],
                       fg=COLOURS["label_fg"], font=FONTS["label"],
                       width=24, anchor="w")
        lbl.pack(side="left")
        widget_frame = tk.Frame(row, bg=COLOURS["panel"])
        widget_frame.pack(side="left", fill="x", expand=True)
        widget.config(bg=COLOURS["entry_bg"],
                      font=FONTS["entry"])
        widget.pack(fill="x")
        lbl._text_key = label_key
        self._lang_widgets.append(lbl)
        return lbl

    def _build_left(self, parent):
        self._lang_widgets = []

        # ── Patient Details card ──────────────────────────────────
        c1 = self._card(parent)
        self._section_label(c1, "patient")

        self.var_name = tk.StringVar()
        ent_name = tk.Entry(c1, textvariable=self.var_name,
                            relief="flat", bd=4)
        self._field_row(c1, "name", ent_name)

        self.var_age = tk.StringVar()
        ent_age = tk.Entry(c1, textvariable=self.var_age,
                           relief="flat", bd=4, width=8)
        self._field_row(c1, "age", ent_age)

        self.var_gender = tk.StringVar(value="Male")
        frame_gender = tk.Frame(c1, bg=COLOURS["panel"])
        self.rb_m = tk.Radiobutton(frame_gender, text="",
                                   variable=self.var_gender, value="Male",
                                   bg=COLOURS["panel"], font=FONTS["label"])
        self.rb_f = tk.Radiobutton(frame_gender, text="",
                                   variable=self.var_gender, value="Female",
                                   bg=COLOURS["panel"], font=FONTS["label"])
        self.rb_m.pack(side="left")
        self.rb_f.pack(side="left", padx=(10, 0))
        self._field_row(c1, "gender", frame_gender)
        tk.Frame(c1, bg=COLOURS["panel"], height=6).pack()

        # ── Vital Signs card ──────────────────────────────────────
        c2 = self._card(parent)
        self._section_label(c2, "vitals")

        self.var_spo2  = tk.StringVar()
        self.var_sbp   = tk.StringVar()
        self.var_dbp   = tk.StringVar()
        self.var_temp  = tk.StringVar()
        self.var_hr    = tk.StringVar()

        for key, var, hint in [
            ("spo2", self.var_spo2, "e.g. 97"),
            ("sbp",  self.var_sbp,  "e.g. 120"),
            ("dbp",  self.var_dbp,  "e.g. 80"),
            ("temp", self.var_temp, "e.g. 37.2"),
            ("hr",   self.var_hr,   "e.g. 80"),
        ]:
            ent = tk.Entry(c2, textvariable=var, relief="flat", bd=4)
            self._field_row(c2, key, ent)
            ent.insert(0, hint)
            ent.config(fg="grey")
            ent.bind("<FocusIn>",  lambda e, v=var, h=hint: self._clear_hint(e, v, h))
            ent.bind("<FocusOut>", lambda e, v=var, h=hint: self._restore_hint(e, v, h))

        tk.Frame(c2, bg=COLOURS["panel"], height=6).pack()

        # ── Complaint card ────────────────────────────────────────
        c3 = self._card(parent)
        self._section_label(c3, "complaint")

        self.var_complaint = tk.StringVar()
        self.cmb_complaint = ttk.Combobox(c3, textvariable=self.var_complaint,
                                          state="readonly",
                                          font=FONTS["entry"])
        row = tk.Frame(c3, bg=COLOURS["panel"])
        row.pack(fill="x", padx=14, pady=(4, 10))
        self.cmb_complaint.pack(fill="x", in_=row)

        # ── Action buttons ────────────────────────────────────────
        btn_row = tk.Frame(parent, bg=COLOURS["bg"])
        btn_row.pack(fill="x", pady=(4, 0))

        self.btn_triage = tk.Button(btn_row, text="", font=FONTS["btn"],
                                    bg=COLOURS["btn_bg"], fg=COLOURS["btn_fg"],
                                    relief="flat", cursor="hand2", pady=10,
                                    command=self._run_triage)
        self.btn_clear  = tk.Button(btn_row, text="", font=FONTS["btn"],
                                    bg="#7F8C8D", fg="white",
                                    relief="flat", cursor="hand2", pady=10,
                                    command=self._clear_all)
        self.btn_print  = tk.Button(btn_row, text="", font=FONTS["btn"],
                                    bg="#1A6B9A", fg="white",
                                    relief="flat", cursor="hand2", pady=10,
                                    command=self._print_slip)
        self.btn_triage.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.btn_clear.pack(side="left",  fill="x", expand=True, padx=(0, 4))
        self.btn_print.pack(side="left",  fill="x", expand=True)

    def _build_right(self, parent):
        """Result panel — right side."""
        # Result card
        self.result_card = tk.Frame(parent, bg=COLOURS["panel"],
                                    highlightbackground=COLOURS["border"],
                                    highlightthickness=1)
        self.result_card.pack(fill="x", pady=(0, 10))

        self.lbl_result_header = tk.Label(self.result_card, text="",
                                          bg=COLOURS["accent"], fg="white",
                                          font=FONTS["section"], pady=8)
        self.lbl_result_header.pack(fill="x")

        self.lbl_triage_label = tk.Label(self.result_card, text="—",
                                         bg=COLOURS["panel"],
                                         font=FONTS["result"], pady=18)
        self.lbl_triage_label.pack()

        self.lbl_confidence = tk.Label(self.result_card, text="",
                                       bg=COLOURS["panel"],
                                       font=FONTS["small"],
                                       fg=COLOURS["label_fg"])
        self.lbl_confidence.pack(pady=(0, 12))

        # Flags card
        self.flags_card = tk.Frame(parent, bg=COLOURS["panel"],
                                   highlightbackground=COLOURS["border"],
                                   highlightthickness=1)
        self.flags_card.pack(fill="both", expand=True)

        self.lbl_flags_header = tk.Label(self.flags_card, text="",
                                         bg="#E74C3C", fg="white",
                                         font=FONTS["section"], pady=8)
        self.lbl_flags_header.pack(fill="x")

        self.txt_flags = tk.Text(self.flags_card, font=FONTS["flag"],
                                 bg=COLOURS["panel"], relief="flat",
                                 wrap="word", height=8,
                                 fg=COLOURS["label_fg"],
                                 state="disabled", padx=10, pady=8)
        self.txt_flags.pack(fill="both", expand=True)

        # Model status badge
        status = "✓ GBC Model Loaded" if MODEL_AVAILABLE else "⚡ Rule-based Mode"
        status_colour = "#27AE60" if MODEL_AVAILABLE else "#E67E22"
        tk.Label(parent, text=status, bg=COLOURS["bg"],
                 fg=status_colour, font=FONTS["small"]).pack(pady=(6, 0))

    # ── LANGUAGE ─────────────────────────────────────────────────

    def _apply_language(self):
        t = LANG[self.lang_code]
        self.lbl_title.config(text=t["title"])
        self.lbl_subtitle.config(text=t["subtitle"])
        self.btn_lang.config(text=t["toggle"])
        self.btn_triage.config(text=t["triage_btn"])
        self.btn_clear.config(text=t["clear_btn"])
        self.btn_print.config(text=t["print_btn"])
        self.lbl_result_header.config(text=t["result_lbl"])
        self.lbl_flags_header.config(text=t["flags"])
        self.rb_m.config(text=t["male"])
        self.rb_f.config(text=t["female"])

        # Update all tagged labels
        for w in self._lang_widgets:
            if hasattr(w, "_text_key"):
                w.config(text=t.get(w._text_key, w._text_key))

        # Update complaint dropdown
        complaint_labels = list(t["complaints"].values())
        self.cmb_complaint.config(values=complaint_labels)
        if not self.var_complaint.get() or self.var_complaint.get() not in complaint_labels:
            self.cmb_complaint.current(0)

    def _toggle_language(self):
        self.lang_code = "ur" if self.lang_code == "en" else "en"
        self._apply_language()

    # ── TRIAGE LOGIC ─────────────────────────────────────────────

    def _run_triage(self):
        try:
            spo2 = float(self.var_spo2.get())
            sbp  = float(self.var_sbp.get())
            dbp  = float(self.var_dbp.get())
            temp = float(self.var_temp.get())
            hr   = float(self.var_hr.get())
        except ValueError:
            messagebox.showerror("Input Error",
                                 "Please enter valid numbers for all vital signs.")
            return

        # Validate ranges
        errors = []
        if not (60 <= spo2 <= 100):  errors.append("SpO₂ must be 60–100 %")
        if not (60 <= sbp  <= 250):  errors.append("Systolic BP must be 60–250 mmHg")
        if not (40 <= dbp  <= 150):  errors.append("Diastolic BP must be 40–150 mmHg")
        if not (34 <= temp <= 43):   errors.append("Temperature must be 34–43 °C")
        if not (30 <= hr   <= 220):  errors.append("Heart Rate must be 30–220 bpm")
        if errors:
            messagebox.showerror("Validation Error", "\n".join(errors))
            return

        # Map selected complaint to code
        t = LANG[self.lang_code]
        selected_label = self.var_complaint.get()
        complaint_map  = {v: k for k, v in t["complaints"].items()}
        complaint_key  = complaint_map.get(selected_label, "general_weakness")
        complaint_code = COMPLAINTS[complaint_key]

        # Run inference
        if MODEL_AVAILABLE:
            result = predict_triage(spo2, sbp, dbp, temp, hr, complaint_code)
        else:
            result = rule_based_triage(spo2, sbp, dbp, temp, hr, complaint_code)

        self.last_result = result
        self._show_result(result)

    def _show_result(self, result):
        label = result["label"]
        conf  = result["confidence"]
        flags = result["flags"]
        colour = COLOURS.get(label, COLOURS["accent"])

        # Triage label
        self.lbl_triage_label.config(text=label, fg=colour)
        self.result_card.config(highlightbackground=colour,
                                highlightthickness=2)
        self.lbl_result_header.config(bg=colour)
        self.lbl_confidence.config(
            text=f"Confidence: {conf:.1%}")

        # Flags
        self.txt_flags.config(state="normal")
        self.txt_flags.delete("1.0", "end")
        for f in flags:
            self.txt_flags.insert("end", f"• {f}\n")
        self.txt_flags.config(state="disabled")

    def _clear_all(self):
        for var in (self.var_name, self.var_age, self.var_spo2,
                    self.var_sbp, self.var_dbp, self.var_temp, self.var_hr):
            var.set("")
        self.var_gender.set("Male")
        self.cmb_complaint.current(0)
        self.lbl_triage_label.config(text="—", fg=COLOURS["label_fg"])
        self.lbl_confidence.config(text="")
        self.txt_flags.config(state="normal")
        self.txt_flags.delete("1.0", "end")
        self.txt_flags.config(state="disabled")
        self.result_card.config(highlightbackground=COLOURS["border"],
                                highlightthickness=1)
        self.lbl_result_header.config(bg=COLOURS["accent"])
        self.last_result = None

    # ── PRINT SLIP ────────────────────────────────────────────────

    def _print_slip(self):
        if not self.last_result:
            messagebox.showinfo("No Result", "Run triage first before printing.")
            return

        now   = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")
        name  = self.var_name.get() or "Unknown"
        age   = self.var_age.get()  or "N/A"
        gen   = self.var_gender.get()
        label = self.last_result["label"]
        conf  = self.last_result["confidence"]
        flags = self.last_result["flags"]

        slip = f"""
╔══════════════════════════════════════════╗
║          SAATHI — OPD TRIAGE SLIP        ║
╠══════════════════════════════════════════╣
  Date / Time : {now}
  Patient     : {name}   Age: {age}   {gen}
  SpO₂        : {self.var_spo2.get()} %
  BP          : {self.var_sbp.get()}/{self.var_dbp.get()} mmHg
  Temp        : {self.var_temp.get()} °C
  HR          : {self.var_hr.get()} bpm
  Complaint   : {self.var_complaint.get()}
──────────────────────────────────────────
  TRIAGE      : *** {label.upper()} ***
  Confidence  : {conf:.1%}
──────────────────────────────────────────
  CLINICAL FLAGS:
"""
        for f in flags:
            slip += f"  • {f}\n"
        slip += "──────────────────────────────────────────\n"
        slip += "  ⚠ This slip is a decision-SUPPORT tool.\n"
        slip += "    Final management responsibility rests\n"
        slip += "    with the attending physician.\n"
        slip += "╚══════════════════════════════════════════╝\n"

        os.makedirs("slips", exist_ok=True)
        filename = f"slips/triage_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(slip)

        # Show in popup
        pop = tk.Toplevel(self)
        pop.title("Triage Slip Preview")
        pop.geometry("480x420")
        pop.configure(bg=COLOURS["panel"])
        txt = tk.Text(pop, font=("Courier", 10), bg=COLOURS["panel"],
                      relief="flat", padx=12, pady=12)
        txt.insert("1.0", slip)
        txt.config(state="disabled")
        txt.pack(fill="both", expand=True)
        tk.Label(pop, text=f"Saved → {filename}",
                 bg=COLOURS["panel"], fg=COLOURS["accent"],
                 font=FONTS["small"]).pack(pady=4)

    # ── HINT HELPERS ─────────────────────────────────────────────

    def _clear_hint(self, event, var, hint):
        if event.widget.get() == hint:
            event.widget.delete(0, "end")
            event.widget.config(fg="black")

    def _restore_hint(self, event, var, hint):
        if not event.widget.get():
            event.widget.insert(0, hint)
            event.widget.config(fg="grey")


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = SAATHIApp()
    app.mainloop()
