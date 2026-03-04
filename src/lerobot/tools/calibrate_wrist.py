from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.motors.feetech import OperatingMode
import json
import select
import sys
import time

PORT = "/dev/tty.usbmodem5AE60840061"
ROBOT_ID = "bombay_girl_follower_arm"  # use your existing id

robot = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID))
robot.connect(calibrate=False)

try:
	motor = "wrist_roll"
	old_offset = None
	motor_cal = robot.bus.read_calibration()
	current_motor_offset = int(motor_cal[motor].homing_offset)
	if robot.calibration_fpath.exists():
		with open(robot.calibration_fpath) as f:
			calibration_from_disk = json.load(f)
		if motor in calibration_from_disk:
			old_offset = int(calibration_from_disk[motor]["homing_offset"])

	robot.bus.disable_torque([motor])
	robot.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

	if old_offset is None:
		old_offset_text = f"<not found in {robot.calibration_fpath}>"
	else:
		old_offset_text = str(old_offset)

	print("Place wrist_roll at desired center, then press ENTER...")
	print("Live preview every 1s (old from disk, new from current pose):")
	model = robot.bus._get_motor_model(motor)
	max_res = robot.bus.model_resolution_table[model] - 1
	while True:
		positions = robot.bus.sync_read("Present_Position", [motor], normalize=False)
		present_pos = int(positions[motor])
		actual_pos = present_pos + current_motor_offset
		proposed_new_offset = actual_pos - int(max_res / 2)
		offset_delta_vs_current = proposed_new_offset - current_motor_offset
		print(
			f"old_homing_offset={old_offset_text} | "
			f"current_motor_homing_offset={current_motor_offset} | "
			f"new_homing_offset={proposed_new_offset} | "
			f"present_pos={present_pos} | "
			f"actual_pos_estimate={actual_pos} | "
			f"offset_delta_vs_current={offset_delta_vs_current}"
		)
		ready, _, _ = select.select([sys.stdin], [], [], 1.0)
		if ready:
			sys.stdin.readline()
			break
		time.sleep(0.0)

	new_offset = int(robot.bus.set_half_turn_homings([motor])[motor])
	cal = robot.bus.read_calibration()

	if old_offset is None:
		print(f"Old {motor} homing_offset: <not found in {robot.calibration_fpath}>")
	else:
		print(f"Old {motor} homing_offset: {old_offset}")
	print(f"Current motor {motor} homing_offset: {current_motor_offset}")
	print(f"New {motor} homing_offset: {new_offset}")

	confirm = input("Type YES to overwrite calibration with the new homing_offset: ").strip()
	if confirm != "YES":
		print("Calibration not changed. Aborting without saving.")
	else:
		cal[motor].homing_offset = new_offset
		cal[motor].range_min = 0
		cal[motor].range_max = 4095

		robot.bus.write_calibration(cal)
		robot.calibration = cal
		robot._save_calibration()

		print("Updated wrist_roll homing_offset:", new_offset)
		print("Saved to:", robot.calibration_fpath)

finally:
	robot.disconnect()
