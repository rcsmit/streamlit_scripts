import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from PIL import Image


PDF_OUT = "cv.pdf"
JPG_OUT = "cv.jpg"

LEFT_BG = HexColor("#E6E6FF")
TEXT = HexColor("#000000")
ACCENT = HexColor("#2F43FF")

WIDTH, HEIGHT = A4
BOTTOM_MARGIN = 2 * cm
TOP_MARGIN = HEIGHT - 2 * cm

PAGE_NUMBER = 1

from datetime import datetime

def ymd_to_dmy(date_str):
    """
    Convert yyyy-mm-dd → dd-mm-yyyy
    """
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%d-%m-%Y")
def load_resume(path="resume.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def check_page_break(c, y, resume):
    if y < BOTTOM_MARGIN:
        c.showPage()
        draw_full_sidebar(c, resume)
        #  draw_empty_sidebar(c)
        return TOP_MARGIN
    return y

def draw_education(c, data, y):
    x = 1 * cm

    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(TEXT)
    c.drawString(x, y, "Education")
    y -= 0.45 * cm

    for edu in data.get("education", []):
        y = check_page_break(c, y, data)
        x = 1 * cm
        c.setFont("Helvetica-Bold", 9)
        c.drawString(
            x,
            y,
            f'{edu["area"]} ({edu["studyType"]})',
        )
        y -= 0.5 * cm

        y = check_page_break(c, y, data)
        
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(
            x,
            y,
            f'{edu["institution"]}'
        )
        y -= 0.6 * cm
        c.drawString(
            x,
            y,
            f'{ymd_to_dmy(edu["startDate"])} to {ymd_to_dmy(edu["endDate"])}',
        )
        y -= 0.6 * cm

        c.setFont("Helvetica", 9)
        for course in edu.get("courses", []):
            y = check_page_break(c, y, data)
            
            c.drawString(x + 0.3 * cm, y, f"{course}")
            y -= 0.45 * cm

        y -= 0.8 * cm

    return y


def draw_full_sidebar(c, data):
    c.setFillColor(LEFT_BG)
    c.rect(0, 0, 7 * cm, HEIGHT, stroke=0, fill=1)

    y = HEIGHT - 2 * cm

    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * cm, y, data["basics"]["name"])
    y -= 1.2 * cm

    c.setFont("Helvetica", 9)
    c.drawString(1 * cm, y, ymd_to_dmy(data["basics"]["birthDate"]))
    y -= 0.6 * cm
    c.drawString(1 * cm, y, data["basics"].get("nationality", ""))
    y -= 1.2 * cm

    y = draw_education(c, data, y)

    def section(title, items):
        nonlocal y
        c.setFont("Helvetica-Bold", 11)
        c.drawString(1 * cm, y, title)
        y -= 0.6 * cm
        c.setFont("Helvetica", 9)
        for i in items:
            c.drawString(1.2 * cm, y, f"• {i}")
            y -= 0.45 * cm
        y -= 0.6 * cm

    
    section(
        "Languages",
        [f'{l["language"]} ({l["fluency"]})' for l in data["languages"]],
    )

    section(
        "Skills",
        [s["name"] for s in data["skills"]],
    )

def draw_empty_sidebar(c):
    c.setFillColor(LEFT_BG)
    c.rect(0, 0, 7 * cm, HEIGHT, stroke=0, fill=1)



def new_page(c, resume):
    global PAGE_NUMBER
    c.showPage()
    PAGE_NUMBER += 1
    print ("PAGE NUMBER:", PAGE_NUMBER)
    if PAGE_NUMBER == 1:
        draw_full_sidebar(c, resume)
    else:
        draw_empty_sidebar(c)

    return HEIGHT - 2 * cm

def check_page_break(c, y, resume):
    
    if y < 2 * cm:
        
        return new_page(c, resume)
    return y

def content_x():
    return 8 * cm #if PAGE_NUMBER == 1 else 2 * cm

def draw_experience(c, data):
    highlights_shown =[]
    global PAGE_NUMBER

    draw_full_sidebar(c, data)
    x = content_x()
    y = HEIGHT - 2 * cm

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "Experience")
    y -= 1.2 * cm

    for job in data["work"]:
        required_height = estimate_job_height(job)

        # If full job does not fit, force new page BEFORE drawing it
        if y - required_height < 2 * cm:
            y = new_page(c, data)
            x = content_x()

        c.setFont("Helvetica-Bold", 11)
        c.setFillColor(ACCENT)
        c.drawString(x, y, job["position"])
        y -= 0.5 * cm

        c.setFont("Helvetica-Oblique", 9)
        c.setFillColor(TEXT)
        c.drawString(
            x,
            y,
            f'{job["name"]} — {ymd_to_dmy(job["startDate"])} to {ymd_to_dmy(job["endDate"])}',
        )
        y -= 0.6 * cm

        c.drawString(x, y, f'{job["location"]}')
        y -= 0.6 * cm

        for h in job.get("highlights", []):
            if h in highlights_shown:
                continue
            c.setFont("Helvetica", 9)
            c.drawString(x + 0.3 * cm, y, f"• {h}")
            y -= 0.45 * cm
            highlights_shown.append(h)

        y -= 0.8 * cm

def estimate_job_height(job):
    base = (
        0.5 +  # position
        0.6 +  # company + dates
        0.6    # location
    )
    highlights = 0.45 * len(job.get("highlights", []))
    spacing = 0.8
    return (base + highlights + spacing) * cm


def build_pdf(data):
    c = canvas.Canvas(PDF_OUT, pagesize=A4)
    #draw_full_sidebar(c, data)
    draw_experience(c, data)
    c.showPage()
    c.save()


def pdf_to_jpg(pdf_path, jpg_path):
    from reportlab.pdfbase import pdfdoc

    img = Image.open(pdf_path)
    img = img.convert("RGB")
    img.save(jpg_path, "JPEG", quality=95)


if __name__ == "__main__":
    url=r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\cv_rene_smit_json_resume.json"
    resume = load_resume(url)
    build_pdf(resume)

    try:
        pdf_to_jpg(PDF_OUT, JPG_OUT)
    except Exception:
        print("JPEG conversion skipped. PDF created successfully.")
