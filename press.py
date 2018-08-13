import win32com.client as comctl
wsh = comctl.Dispatch("WScript.Shell")

# Google Chrome window title
wsh.AppActivate("icanhazip.com")
wsh.SendKeys("A")
