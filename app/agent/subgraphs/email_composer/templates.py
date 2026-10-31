"""
app/agent/subgraphs/email_composer/templates.py
===============================================
Production-grade responsive HTML email renderer for Vyapar Sathi.
Uses inline CSS and bulletproof email design principles.
"""

from __future__ import annotations
import html
from typing import Dict, Any
from app.agent.subgraphs.email_composer.schemas import EmailDraftSpec, EmailCategory


_CATEGORY_THEMES = {
    "purchase_order": {
        "primary": "#2563eb",      # Blue 600
        "primary_dark": "#1d4ed8", # Blue 700
        "badge_bg": "#eff6ff",     # Blue 50
        "badge_text": "#1d4ed8",   # Blue 700
        "border_color": "#bfdbfe", # Blue 200
        "accent": "#3b82f6",
        "icon": "📦",
    },
    "payment_reminder": {
        "primary": "#d97706",      # Amber 600
        "primary_dark": "#b45309", # Amber 700
        "badge_bg": "#fffbeb",     # Amber 50
        "badge_text": "#b45309",   # Amber 700
        "border_color": "#fde68a", # Amber 200
        "accent": "#f59e0b",
        "icon": "💳",
    },
    "quotation_inquiry": {
        "primary": "#0891b2",      # Cyan 600
        "primary_dark": "#0e7490", # Cyan 700
        "badge_bg": "#ecfeff",     # Cyan 50
        "badge_text": "#0e7490",   # Cyan 700
        "border_color": "#a5f3fc", # Cyan 200
        "accent": "#06b6d4",
        "icon": "📋",
    },
    "restock_alert": {
        "primary": "#e11d48",      # Rose 600
        "primary_dark": "#be123c", # Rose 700
        "badge_bg": "#fff1f2",     # Rose 50
        "badge_text": "#be123c",   # Rose 700
        "border_color": "#fecdd3", # Rose 200
        "accent": "#f43f5e",
        "icon": "⚠️",
    },
    "customer_announcement": {
        "primary": "#7c3aed",      # Violet 600
        "primary_dark": "#6d28d9", # Violet 700
        "badge_bg": "#f5f3ff",     # Violet 50
        "badge_text": "#6d28d9",   # Violet 700
        "border_color": "#ddd6fe", # Violet 200
        "accent": "#8b5cf6",
        "icon": "🎉",
    },
    "inventory_report": {
        "primary": "#059669",      # Emerald 600
        "primary_dark": "#047857", # Emerald 700
        "badge_bg": "#ecfdf5",     # Emerald 50
        "badge_text": "#047857",   # Emerald 700
        "border_color": "#a7f3d0", # Emerald 200
        "accent": "#10b981",
        "icon": "📊",
    },
    "sales_summary": {
        "primary": "#4f46e5",      # Indigo 600
        "primary_dark": "#4338ca", # Indigo 700
        "badge_bg": "#eef2ff",     # Indigo 50
        "badge_text": "#4338ca",   # Indigo 700
        "border_color": "#c7d2fe", # Indigo 200
        "accent": "#6366f1",
        "icon": "📈",
    },
    "general_correspondence": {
        "primary": "#0f172a",      # Slate 900
        "primary_dark": "#020617", # Slate 950
        "badge_bg": "#f1f5f9",     # Slate 100
        "badge_text": "#334155",   # Slate 700
        "border_color": "#cbd5e1", # Slate 300
        "accent": "#475569",
        "icon": "✉️",
    },
}


def render_html_email(spec: EmailDraftSpec, category: str = "general_correspondence") -> str:
    """
    Renders an EmailDraftSpec into a mobile-first, responsive, styled HTML email.
    """
    theme = _CATEGORY_THEMES.get(category, _CATEGORY_THEMES["general_correspondence"])

    # 1. Summary Cards HTML
    summary_cards_html = ""
    if spec.summary_cards:
        cards = []
        for c in spec.summary_cards:
            c_label = html.escape(c.label)
            c_val = html.escape(c.value)
            cards.append(f"""
            <td style="padding: 6px; width: 33.33%; vertical-align: top;">
              <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 12px; text-align: center;">
                <div style="font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 3px;">{c_label}</div>
                <div style="font-size: 15px; font-weight: 700; color: #0f172a;">{c_val}</div>
              </div>
            </td>
            """)
        summary_cards_html = f"""
        <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="margin: 18px 0;">
          <tr>
            {''.join(cards)}
          </tr>
        </table>
        """

    # 2. Table HTML
    table_html = ""
    if spec.table_headers and spec.table_rows:
        header_th = "".join(
            f'<th style="padding: 10px 12px; font-size: 12px; font-weight: 700; color: #334155; text-align: {"right" if i > 0 and ("rate" in h.lower() or "price" in h.lower() or "total" in h.lower() or "subtotal" in h.lower()) else "left"}; background: #f1f5f9; border-bottom: 2px solid #cbd5e1;">{html.escape(h)}</th>'
            for i, h in enumerate(spec.table_headers)
        )

        rows_tr = []
        for row_idx, row in enumerate(spec.table_rows):
            bg = "#ffffff" if row_idx % 2 == 0 else "#f8fafc"
            cells = []
            for col_idx, cell in enumerate(row):
                is_num = col_idx > 0 and any(kw in spec.table_headers[col_idx].lower() for kw in ("rate", "price", "total", "subtotal", "qty", "quantity"))
                align = "right" if is_num else "left"
                cells.append(
                    f'<td style="padding: 9px 12px; font-size: 13px; color: #1e293b; border-bottom: 1px solid #e2e8f0; text-align: {align};">{html.escape(str(cell))}</td>'
                )
            rows_tr.append(f'<tr style="background: {bg};">{"".join(cells)}</tr>')

        # Table Footer
        footer_tr = ""
        if spec.table_footer:
            f_rows = []
            for f_key, f_val in spec.table_footer.items():
                f_rows.append(f"""
                <tr>
                  <td colspan="{len(spec.table_headers)-1}" style="padding: 8px 12px; font-size: 13px; font-weight: 700; text-align: right; color: #334155; border-top: 1px solid #cbd5e1;">{html.escape(f_key)}:</td>
                  <td style="padding: 8px 12px; font-size: 14px; font-weight: 800; text-align: right; color: {theme['primary']}; border-top: 1px solid #cbd5e1;">{html.escape(f_val)}</td>
                </tr>
                """)
            footer_tr = "".join(f_rows)

        table_html = f"""
        <div style="margin: 20px 0; overflow-x: auto; border: 1px solid #e2e8f0; border-radius: 8px;">
          <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="border-collapse: collapse; width: 100%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
            <thead>
              <tr>{header_th}</tr>
            </thead>
            <tbody>
              {''.join(rows_tr)}
              {footer_tr}
            </tbody>
          </table>
        </div>
        """

    # 3. Intro Paragraphs
    intro_html = "".join(
        f'<p style="margin: 0 0 12px 0; font-size: 14px; line-height: 1.6; color: #334155;">{html.escape(p)}</p>'
        for p in spec.intro_paragraphs
    )

    # 4. Instruction Paragraphs
    instruction_html = ""
    if spec.instruction_paragraphs:
        inst_p = "".join(
            f'<li style="margin-bottom: 6px; font-size: 13px; line-height: 1.5; color: #475569;">{html.escape(p)}</li>'
            for p in spec.instruction_paragraphs
        )
        instruction_html = f"""
        <div style="margin: 18px 0; padding: 14px 16px; background: #faf5ff; border-left: 4px solid {theme['primary']}; border-radius: 4px;">
          <div style="font-size: 13px; font-weight: 700; color: #1e1b4b; margin-bottom: 6px;">Important Instructions & Notes:</div>
          <ul style="margin: 0; padding-left: 18px;">
            {inst_p}
          </ul>
        </div>
        """

    # 5. Action Button
    cta_html = ""
    if spec.action_button and spec.action_button.text:
        b_text = html.escape(spec.action_button.text)
        b_url = html.escape(spec.action_button.url or "#")
        cta_html = f"""
        <div style="text-align: center; margin: 24px 0 16px 0;">
          <a href="{b_url}" style="display: inline-block; padding: 12px 28px; background-color: {theme['primary']}; color: #ffffff; text-decoration: none; font-size: 14px; font-weight: bold; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">{b_text}</a>
        </div>
        """

    # 6. Sender Signature
    s = spec.sender
    phone_html = f'<div><b>Phone:</b> <a href="tel:{html.escape(s.phone)}" style="color: #475569; text-decoration: none;">{html.escape(s.phone)}</a></div>' if s.phone else ""
    gstin_html = f'<div><b>GSTIN:</b> {html.escape(s.gstin)}</div>' if s.gstin else ""
    address_html = f'<div><b>Location:</b> {html.escape(s.address)}</div>' if s.address else ""

    signature_html = f"""
    <div style="margin-top: 28px; padding-top: 18px; border-top: 1px solid #e2e8f0; font-size: 13px; color: #475569; line-height: 1.6;">
      <div style="font-weight: 700; font-size: 14px; color: #0f172a; margin-bottom: 3px;">{html.escape(s.store_name)}</div>
      <div style="color: #334155; margin-bottom: 4px;"><b>Merchant / Owner:</b> {html.escape(s.owner_name)}</div>
      <div><b>Official Email:</b> <a href="mailto:{html.escape(s.email)}" style="color: {theme['primary']}; text-decoration: none;">{html.escape(s.email)}</a></div>
      {phone_html}
      {gstin_html}
      {address_html}
    </div>
    """

    # Assemble Full Document
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(spec.subject)}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
  </style>
</head>
<body style="margin: 0; padding: 0; background-color: #f1f5f9; font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; -webkit-font-smoothing: antialiased;">
  <!-- Preview snippet for email clients -->
  <div style="display: none; max-height: 0px; overflow: hidden; opacity: 0;">
    {html.escape(spec.preview_text)}
  </div>

  <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="background-color: #f1f5f9; padding: 32px 12px;">
    <tr>
      <td align="center">
        <!-- Main Card Container -->
        <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="max-width: 600px; background-color: #ffffff; border-radius: 16px; overflow: hidden; box-shadow: 0 10px 30px -5px rgba(15, 23, 42, 0.08), 0 0 0 1px rgba(226, 232, 240, 0.8);">
          
          <!-- Top Header Bar -->
          <tr>
            <td style="background: linear-gradient(135deg, {theme['primary_dark']}, {theme['primary']}); padding: 32px 32px 28px 32px; text-align: left;">
              <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%">
                <tr>
                  <td>
                    <span style="display: inline-block; font-size: 11px; font-weight: 800; letter-spacing: 0.08em; color: #ffffff; text-transform: uppercase; background: rgba(255,255,255,0.22); padding: 4px 12px; border-radius: 9999px;">
                      {theme['icon']} {html.escape(spec.badge_title)}
                    </span>
                    <h1 style="margin: 12px 0 0 0; font-size: 22px; font-weight: 800; color: #ffffff; line-height: 1.3; letter-spacing: -0.02em;">
                      {html.escape(spec.headline)}
                    </h1>
                  </td>
                  <td align="right" style="vertical-align: top;">
                    <div style="font-size: 13px; font-weight: 700; color: #e0e7ff; background: rgba(0,0,0,0.18); padding: 4px 12px; border-radius: 8px;">
                      {html.escape(spec.sender.store_name)}
                    </div>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- Email Content Body -->
          <tr>
            <td style="padding: 32px 32px 28px 32px;">
              <!-- Salutation -->
              <p style="margin: 0 0 14px 0; font-size: 15px; font-weight: 700; color: #0f172a;">
                {html.escape(spec.salutation)}
              </p>

              <!-- Intro Paragraphs -->
              {intro_html}

              <!-- Summary Cards Grid -->
              {summary_cards_html}

              <!-- Line Items Table -->
              {table_html}

              <!-- Special Instructions -->
              {instruction_html}

              <!-- Action Callout Button -->
              {cta_html}

              <!-- Closing Remark -->
              <p style="margin: 20px 0 0 0; font-size: 14px; font-weight: 600; color: #334155;">
                {html.escape(spec.closing_text)}
              </p>

              <!-- Sender Signature Box -->
              {signature_html}
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="background-color: #f8fafc; padding: 20px 24px; text-align: center; font-size: 12px; color: #94a3b8; border-top: 1px solid #f1f5f9;">
              Dispatched with precision via <b>Vyapar Sakha AI</b> for {html.escape(spec.sender.store_name)}.
              <div style="margin-top: 4px; font-size: 11px; color: #cbd5e1;">© 2026 Vyapar Sakha • Smart Store Automation</div>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""
    return full_html


def render_plain_text(spec: EmailDraftSpec) -> str:
    """Renders a clean plain-text fallback version."""
    lines = [
        f"{spec.badge_title}: {spec.headline}",
        f"From: {spec.sender.store_name}",
        "=" * 40,
        "",
        spec.salutation,
        "",
    ]
    for p in spec.intro_paragraphs:
        lines.append(p)
        lines.append("")

    if spec.summary_cards:
        for c in spec.summary_cards:
            lines.append(f"• {c.label}: {c.value}")
        lines.append("")

    if spec.table_headers and spec.table_rows:
        lines.append("ITEMS:")
        lines.append(" - " + " | ".join(spec.table_headers))
        for row in spec.table_rows:
            lines.append(" - " + " | ".join(row))
        if spec.table_footer:
            for k, v in spec.table_footer.items():
                lines.append(f"{k}: {v}")
        lines.append("")

    if spec.instruction_paragraphs:
        lines.append("INSTRUCTIONS:")
        for p in spec.instruction_paragraphs:
            lines.append(f"- {p}")
        lines.append("")

    lines.append(spec.closing_text)
    lines.append("")
    lines.append(f"---\n{spec.sender.store_name}\nOwner: {spec.sender.owner_name}\nEmail: {spec.sender.email}")
    if spec.sender.phone:
        lines.append(f"Phone: {spec.sender.phone}")
    if spec.sender.gstin:
        lines.append(f"GSTIN: {spec.sender.gstin}")

    return "\n".join(lines)
