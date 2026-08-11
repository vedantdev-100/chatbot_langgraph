import subprocess

result = subprocess.run(
    [
        r"C:\Users\Pro-3\AppData\Local\Programs\Python\Python311\Scripts\uv.exe",
        "--directory", r"C:\Users\Pro-3\Desktop\local_mcp_lgin",
        "run", "main.py",
    ],
    input=b"",  # send empty input so it doesn't hang waiting on stdin
    capture_output=True,
    timeout=5,
)
print("STDOUT:", result.stdout.decode(errors="replace"))
print("STDERR:", result.stderr.decode(errors="replace"))
print("RETURN CODE:", result.returncode)