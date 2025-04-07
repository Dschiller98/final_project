import subprocess

def launch(script_path):
    """
    Run a test script and print its output.
    Args:
        script_path (str): Path to the test script to execute.
    """
    print(f"Running {script_path}...")
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_path}: {e}")

if __name__ == "__main__":
    # List of test scripts to execute
    test_scripts = [
        "irobman-wise-2425-final-project/test_perception.py",
        "irobman-wise-2425-final-project/test_control.py",
        "irobman-wise-2425-final-project/test_grasping.py",
        "irobman-wise-2425-final-project/test_pnp.py",
        "irobman-wise-2425-final-project/test_tracking.py",
        "irobman-wise-2425-final-project/main.py"
    ]

    # Execute each test script
    for script in test_scripts:
        launch(script)