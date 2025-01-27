duration_ms_default = 1000
duration_ms = int(input(
    f'Press RETURN to open the valve for {duration_ms_default} ms or enter a different value').strip() or duration_ms_default)
print(duration_ms)