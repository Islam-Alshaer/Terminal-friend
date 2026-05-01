import os
import platform

def get_os_details() -> str:
    """returns a dictionary of OS details like: name, version and distro"""
    system = platform.system()

    if "linux" in system.lower():
        return str(get_linux_details())
    else:
        return str(platform.uname())[13:-1]


def get_linux_details() -> dict:
    """Parse /etc/os-release (or fallback files) for Linux distro info"""
    os_info = {}
    paths = [
        "/etc/os-release",
        "/usr/lib/os-release",
        "/etc/lsb-release"  # Legacy Ubuntu/Debian
    ]

    for path in paths:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        k = k.lower() #to prevent case problems
                        # Remove quotes: "Ubuntu 22.04" → Ubuntu 22.04
                        os_info[k] = v.strip('"\'')
            break

    return os_info

if __name__ == "__main__":
    print(get_os_details())
