import { useState, useEffect, useCallback, useRef } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  faArrowRotateLeft,
  faArrowRotateRight
} from '@fortawesome/free-solid-svg-icons'

import config from "./config.js";
import JoyStick from "./joy";

/********************
 * MOVE PANEL
 ********************/

function DriveControlPanel({ drivingMode, mbot }) {
  const [speed, setSpeed] = useState(50);
  const [joyActive, setJoyActive] = useState(false);
  const [keyActive, setKeyActive] = useState(false);

  const driveCmd = useRef({ vx: 0, vy: 0, wz: 0 });
  const controlMapRef = useRef({
    s: { pressed: false, fn: "back" },
    w: { pressed: false, fn: "forward" },
    a: { pressed: false, fn: "tleft" },
    d: { pressed: false, fn: "tright" },
    e: { pressed: false, fn: "right" },
    q: { pressed: false, fn: "left" },
  });
  const keyPressRef = useRef({ x: 0, y: 0, t: 0 });

  const handleKeyDown = useCallback((evt) => {
    let controlMap = controlMapRef.current;
    let keyPress = keyPressRef.current;
    if (drivingMode && controlMap[evt.key]) {
      if (!keyActive) setKeyActive(true);

      controlMap[evt.key].pressed = true;
      if (controlMap[evt.key].fn === "back" && keyPress.x > -1) keyPress.x--;
      if (controlMap[evt.key].fn === "forward" && keyPress.x < 1) keyPress.x++;
      if (controlMap[evt.key].fn === "right" && keyPress.y > -1) keyPress.y--;
      if (controlMap[evt.key].fn === "left" && keyPress.y < 1) keyPress.y++;
      if (controlMap[evt.key].fn === "tright" && keyPress.t > -1) keyPress.t--;
      if (controlMap[evt.key].fn === "tleft" && keyPress.t < 1) keyPress.t++;

      driveCmd.current.vx = keyPress.x * speed / 100;
      driveCmd.current.vy = keyPress.y * speed / 100;
      driveCmd.current.wz = config.ANG_VEL_MULTIPLIER * speed * keyPress.t / 100;
    }
  }, [drivingMode, keyActive, setKeyActive, controlMapRef, keyPressRef, speed]);

  const handleKeyUp = useCallback((evt) => {
    let controlMap = controlMapRef.current;
    let keyPress = keyPressRef.current;
    if (drivingMode && controlMap[evt.key]) {
      controlMap[evt.key].pressed = false;
      if (controlMap[evt.key].fn === "back" && keyPress.x < 1) keyPress.x++;
      if (controlMap[evt.key].fn === "forward" && keyPress.x > -1) keyPress.x--;
      if (controlMap[evt.key].fn === "right" && keyPress.y < 1) keyPress.y++;
      if (controlMap[evt.key].fn === "left" && keyPress.y > -1) keyPress.y--;
      if (controlMap[evt.key].fn === "tright" && keyPress.t < 1) keyPress.t++;
      if (controlMap[evt.key].fn === "tleft" && keyPress.t > -1) keyPress.t--;

      let reset = true;
      for (const key in controlMap) {
        if (controlMap[key].pressed) reset = false;
      }
      if (reset) {
        keyPress.x = 0;
        keyPress.y = 0;
        keyPress.t = 0;
        if (keyActive) setKeyActive(false);
      }

      driveCmd.current.vx = keyPress.x * speed / 100;
      driveCmd.current.vy = keyPress.y * speed / 100;
      driveCmd.current.wz = config.ANG_VEL_MULTIPLIER * speed * keyPress.t / 100;
    }
  }, [drivingMode, keyActive, setKeyActive, controlMapRef, keyPressRef, speed]);

  const handleJoystickMove = useCallback((stickData) => {
    if (keyActive) return;  // Ignore joystick if we are using the keyboard keys.
    if (stickData.active !== joyActive) setJoyActive(stickData.active);

    if (!stickData.active) {
      driveCmd.current.vx = 0;
      driveCmd.current.vy = 0;
      driveCmd.current.wz = 0;
    }
    else {
      driveCmd.current.vx = stickData.y * speed / 10000;
      driveCmd.current.vy = 0;
      driveCmd.current.wz = -stickData.x * speed / 10000;
    }
  }, [keyActive, joyActive, setJoyActive, speed]);

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('keyup', handleKeyUp);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('keyup', handleKeyUp);
      mbot.stop();
    };
  }, [handleKeyDown, handleKeyUp, speed]);

  // This effect starts a timer to send commands to the robot IF one of the
  // control mechanisms is active. This allows us to control the frequency of
  // the control commands.
  useEffect(() => {
    let timerId = null;

    if (joyActive || keyActive) {
      mbot.drive(driveCmd.current.vx, driveCmd.current.vy, driveCmd.current.wz);
      timerId = setInterval(() => {
        mbot.drive(driveCmd.current.vx, driveCmd.current.vy, driveCmd.current.wz);
      }, config.DRIVE_CMD_RATE);
    }

    // Return the cleanup function which stops the controller and stops the robot..
    return () => {
      if (timerId) {
        clearInterval(timerId);
        mbot.stop();
      }
    };
  }, [joyActive, keyActive]);

  const joyStyle = {
    internalFillColor: "#1397cf",
    internalStrokeColor: "#2F65A7",
    externalStrokeColor: "#2F65A7"
  };

  return (
    <div className="drive-panel-wrapper">
      <div className="drive-buttons">
        <button className="button drive-turn" id="turn-left"
          onMouseDown={() => mbot.drive(0, 0, config.ANG_VEL_MULTIPLIER * speed / 100)}
          onMouseUp={() => mbot.stop()}>
          <FontAwesomeIcon icon={faArrowRotateLeft} />
        </button>

        <button className="button drive-turn" id="turn-right"
          onMouseDown={() => mbot.drive(0, 0, -config.ANG_VEL_MULTIPLIER * speed / 100)}
          onMouseUp={() => mbot.stop()}>
          <FontAwesomeIcon icon={faArrowRotateRight} />
        </button>
      </div>

      <div id="joy1Div" className={`joyStyle`}>
        <JoyStick containerId="joy1Div" parameters={joyStyle} callback={handleJoystickMove} />
      </div>

      <div className="button-wrapper-row top-spacing">
        <button className="button stop-color col-lg-12" id="drive-stop" onClick={() => mbot.stop()}>Stop</button>
      </div>

      <div className="col-lg-12">
        <span>Speed: {speed} &nbsp;&nbsp;</span>
        <input type="range" min="1" max="100" value={speed} onChange={(event) => setSpeed(event.target.value)}></input>
      </div>
    </div>
  );
}

export { DriveControlPanel };
