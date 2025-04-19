/*
 * Reactified version of the following code.
 *
 * Name          : joy.js
 * @author       : Roberto D'Amico (Bobboteck)
 * Last modified : 09.06.2020
 * Revision      : 1.1.6
 *
 * Modification History:
 * Date         Version     Modified By     Description
 * 2021-12-21   2.0.0       Roberto D'Amico New version of the project that integrates the callback functions, while
 *                                          maintaining compatibility with previous versions. Fixed Issue #27 too,
 *                                          thanks to @artisticfox8 for the suggestion.
 * 2020-06-09   1.1.6       Roberto D'Amico Fixed Issue #10 and #11
 * 2020-04-20   1.1.5       Roberto D'Amico Correct: Two sticks in a row, thanks to @liamw9534 for the suggestion
 * 2020-04-03               Roberto D'Amico Correct: InternalRadius when change the size of canvas, thanks to
 *                                          @vanslipon for the suggestion
 * 2020-01-07   1.1.4       Roberto D'Amico Close #6 by implementing a new parameter to set the functionality of
 *                                          auto-return to 0 position
 * 2019-11-18   1.1.3       Roberto D'Amico Close #5 correct indication of East direction
 * 2019-11-12   1.1.2       Roberto D'Amico Removed Fix #4 incorrectly introduced and restored operation with touch
 *                                          devices
 * 2019-11-12   1.1.1       Roberto D'Amico Fixed Issue #4 - Now JoyStick work in any position in the page, not only
 *                                          at 0,0
 *
 * The MIT License (MIT)
 *
 *  This file is part of the JoyStick Project (https://github.com/bobboteck/JoyStick).
 *	Copyright (c) 2015 Roberto D'Amico (Bobboteck).
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';

const StickStatus = {
  xPosition: 0,
  yPosition: 0,
  x: 0,
  y: 0,
  cardinalDirection: "C",
  active: false
};

const DEFAULT_PARAMS = {
  title: 'joystick',
  width: 0,
  height: 0,
  internalFillColor: "#00AA00",
  internalLineWidth: 2,
  internalStrokeColor: "#003300",
  externalLineWidth: 2,
  externalStrokeColor: "#008000",
  autoReturnToCenter: true
};

const JoyStick = ({ containerId, parameters = {}, callback = () => {} }) => {
  const {
    title,
    width,
    height,
    internalFillColor,
    internalLineWidth,
    internalStrokeColor,
    externalLineWidth,
    externalStrokeColor,
    autoReturnToCenter
  } = { ...DEFAULT_PARAMS, ...parameters };

  const canvasRef = useRef(null);
  const contextRef = useRef(null);
  const internalRadiusRef = useRef(0);
  const maxMoveStickRef = useRef(0);
  const externalRadiusRef = useRef(0);
  const [centerX, setCenterX] = useState(0);
  const [centerY, setCenterY] = useState(0);
  const [moved, setMoved] = useState({x: 0, y: 0});
  const [pressed, setPressed] = useState(false);

  const circumference = 2 * Math.PI;

  const drawExternal = useCallback((centerX, centerY) => {
    const ctx = contextRef.current;
    if (!ctx) return;

    ctx.beginPath();
    ctx.arc(centerX, centerY, externalRadiusRef.current, 0, circumference, false);
    ctx.lineWidth = externalLineWidth;
    ctx.strokeStyle = externalStrokeColor;
    ctx.stroke();
  }, [contextRef, externalRadiusRef]);

  const drawInternal = useCallback((movedX, movedY) => {
    const ctx = contextRef.current;
    if (!ctx) return;

    const internalRadius = internalRadiusRef.current;
    ctx.beginPath();
    const movedXAdjusted = Math.max(internalRadius, Math.min(movedX, canvasRef.current.width - internalRadius));
    const movedYAdjusted = Math.max(internalRadius, Math.min(movedY, canvasRef.current.height - internalRadius));
    ctx.arc(movedXAdjusted, movedYAdjusted, internalRadius, 0, circumference, false);
    const grd = ctx.createRadialGradient(centerX, centerY, 5, centerX, centerY, 200);
    grd.addColorStop(0, internalFillColor);
    grd.addColorStop(1, internalStrokeColor);
    ctx.fillStyle = grd;
    ctx.fill();
    ctx.lineWidth = internalLineWidth;
    ctx.strokeStyle = internalStrokeColor;
    ctx.stroke();
  }, [contextRef, internalRadiusRef, centerX, centerY]);

  const getCardinalDirection = useCallback((x, y) => {
    const horizontal = x - centerX;
    const vertical = y - centerY;
    const directionHorizontalLimitPos = canvasRef.current.width / 10;
    const directionHorizontalLimitNeg = directionHorizontalLimitPos * -1;
    const directionVerticalLimitPos = canvasRef.current.height / 10;
    const directionVerticalLimitNeg = directionVerticalLimitPos * -1;

    let result = "";
    if (vertical >= directionVerticalLimitNeg && vertical <= directionVerticalLimitPos) {
      result = "C";
    }
    if (vertical < directionVerticalLimitNeg) {
      result = "N";
    }
    if (vertical > directionVerticalLimitPos) {
      result = "S";
    }
    if (horizontal < directionHorizontalLimitNeg) {
      result += result === "C" ? "W" : "W";
    }
    if (horizontal > directionHorizontalLimitPos) {
      result += result === "C" ? "E" : "E";
    }
    return result;
  }, [centerX, centerY, canvasRef]);

  const updateStickStatus = useCallback((x, y, active = true) => {
    const status = {
      xPosition: x,
      yPosition: y,
      active: active,
      x: (100 * ((x - centerX) / maxMoveStickRef.current)).toFixed(),
      y: ((100 * ((y - centerY) / maxMoveStickRef.current)) * -1).toFixed(),
      cardinalDirection: getCardinalDirection(x, y),
    }
    callback(status);
  }, [centerX, centerY, maxMoveStickRef, callback, getCardinalDirection]);

  const onTouchStart = useCallback(() => {
    setPressed(true);
  }, [setPressed]);

  const onTouchMove = useCallback((event) => {
    if (pressed) {
      event.preventDefault(); // Prevent scrolling
      const touch = event.targetTouches[0];
      const canvas = canvasRef.current;
      if (touch.target === canvas) {
        const newX = touch.pageX - canvas.offsetLeft;
        const newY = touch.pageY - canvas.offsetTop;
        setMoved({x: newX, y: newY});
        updateStickStatus(newX, newY, true);
      }
    }
  }, [pressed, setMoved, updateStickStatus, canvasRef]);

  const onTouchEnd = useCallback(() => {
    console.log("touch up")
    setPressed(false);
    if (autoReturnToCenter) {
      setMoved({x: centerX, y: centerY});
    }
    updateStickStatus(centerX, centerY, false);
  }, [setPressed, setMoved, centerX, centerY, updateStickStatus]);

  const onMouseDown = useCallback(() => {
    setPressed(true);
  }, [setPressed]);

  const onMouseMove = useCallback((event) => {
    if (pressed) {
      const canvas = canvasRef.current;
      const newX = event.pageX - canvas.offsetLeft;
      const newY = event.pageY - canvas.offsetTop;
      setMoved({x: newX, y: newY});
      // setMovedY(newY);
      updateStickStatus(newX, newY, true);
    }
  }, [pressed, setMoved, updateStickStatus, canvasRef]);

  const onMouseUp = useCallback(() => {
    setPressed(false);
    if (autoReturnToCenter) {
      setMoved({x: centerX, y: centerY});
    }
    updateStickStatus(centerX, centerY, false);
  }, [setPressed, setMoved, centerX, centerY, updateStickStatus]);

  useEffect(() => {
    const objContainer = document.getElementById(containerId);
    objContainer.style.touchAction = "none"; // Prevent default touch action
    const canvas = canvasRef.current;
    canvas.width = width > 0 ? width : objContainer.clientWidth;
    canvas.height =  height > 0 ? height : objContainer.clientHeight;
    const ctx = canvas.getContext('2d');
    contextRef.current = ctx;

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    setCenterX(centerX);
    setCenterY(centerY);
    setMoved({x: centerX, y: centerY});

    const internalRadius = (canvas.width - ((canvas.width / 2) + 10)) / 2;
    const maxMoveStick = internalRadius + 5;
    const externalRadius = internalRadius + 30;

    internalRadiusRef.current = internalRadius;
    maxMoveStickRef.current = maxMoveStick;
    externalRadiusRef.current = externalRadius;

    // Check if the device supports touch
    if ("ontouchstart" in document.documentElement) {
      canvas.addEventListener('touchstart', onTouchStart);
      document.addEventListener('touchmove', onTouchMove);
      document.addEventListener('touchend', onTouchEnd);
    } else {
      canvas.addEventListener('mousedown', onMouseDown);
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    }

    drawExternal(centerX, centerY);
    drawInternal(centerX, centerY);

    return () => {
      canvas.removeEventListener('touchstart', onTouchStart);
      document.removeEventListener('touchmove', onTouchMove);
      document.removeEventListener('touchend', onTouchEnd);
      canvas.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
  }, [containerId, canvasRef, drawExternal, drawInternal,
      setCenterX, setCenterY, setMoved,
      onTouchStart, onTouchMove, onTouchEnd,
      onMouseDown, onMouseMove, onMouseUp]);

  useEffect(() => {
    if (contextRef.current) {
      contextRef.current.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      drawExternal(centerX, centerY);
      drawInternal(moved.x, moved.y);
    }
  }, [moved, centerX, centerY, drawExternal, drawInternal, contextRef, canvasRef]);

  return (
    <canvas id={title} ref={canvasRef} style={{ touchAction: 'none' }} />
  );
};

export default JoyStick;
