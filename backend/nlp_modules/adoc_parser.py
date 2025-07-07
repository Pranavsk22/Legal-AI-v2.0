import subprocess


ASCIIDOCTOR = r"C:\ProgramData\chocolatey\lib\asciidoctorj\tools\asciidoctorj-2.5.13\bin\asciidoctorj.bat"

def extract_text_from_adoc(path):
    result = subprocess.run(
        [ASCIIDOCTOR, "-b", "docbook", "-o", "-", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",  # 🔥 Explicitly set UTF-8 decoding
        errors="replace",  # 🔧 Avoid crashing on weird characters
    )
    return result.stdout
