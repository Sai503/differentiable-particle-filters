import { useState, useEffect, useRef, useCallback } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faBars, faCircleInfo } from '@fortawesome/free-solid-svg-icons'

import config from "./config.js";
import { DriveControlPanel } from "./driveControls";
import { MBotScene } from './scene.js'
import { downloadMapFile } from "./map.js";

function ConnectionStatus({ status }) {
  let msg = "Wait";
  let colour = "#ffd300";
  if (status === true) {
    msg = "Connected";
    colour = "#00ff00";
  }
  else if (status === false) {
    msg = "Not Connected";
    colour = "#ff0000";
  }

  return (
    <div className="status" style={{backgroundColor: colour}}>
      {msg}
    </div>
  );
}

function StatusMessage({ robotPose, robotCell, clickedCell }) {
  let msg = [];
  if(robotPose != null){
    msg.push(
      <p className="robot-info" key="robotInfoPose">
        <i>Robot Pose:</i> (
          <b>x:</b> {robotPose.x.toFixed(3)},&nbsp;
          <b>y:</b> {robotPose.y.toFixed(3)},&nbsp;
          <b>t:</b> {robotPose.theta.toFixed(3)})
      </p>
    );
    msg.push(
      <p className="robot-info" key="robotInfoCell">
        <i>Robot Cell:</i> ({robotCell[0]}, {robotCell[1]})
      </p>
  );
  }
  if (clickedCell.length > 0) {
    msg.push(
      <p className="robot-info" key="robotInfoClicked">
        <i>Clicked:</i>&nbsp;
        <b>x:</b> {clickedCell[2].toFixed(3)},&nbsp;
        <b>y:</b> {clickedCell[3].toFixed(3)},&nbsp;
        Cell: [{clickedCell[1]}, {clickedCell[0]}]
      </p>
    );
  }

  return (
    <div className="status-msg">
      {msg}
    </div>
  );
}

function ToggleSelect({ small, label, explain, checked, onChange, isActive = true }) {
  const [viewInfo, setViewInfo] = useState(false);
  const [top, setTop] = useState(0);

  let sizeCls = "";
  if (small) sizeCls = " small";

  const toggleClasses = isActive ? "slider round" + sizeCls : "slider round disabled" + sizeCls;

  return (
    <div className="toggle-wrapper">
      <div className="row">
        <div className="col-7">
          <span>{label}</span>
        </div>
        <div
          className="col-1 info"
          onMouseEnter={(evt) => {
            setViewInfo(true);
            setTop(evt.clientY - 20);
          }}
          onMouseLeave={() => { setViewInfo(false); }}
        >
          <div className="info-icon">
            <FontAwesomeIcon icon={faCircleInfo} size="xs" />
          </div>
        </div>
        {viewInfo && (
          <span className="explain" style={{ top: top }}>
            {explain}
          </span>
        )}
        <div className="col-4 text-right toggle">
          <label className={"switch" + sizeCls}>
            <input
              type="checkbox"
              className="mx-2"
              checked={checked}
              onChange={isActive ? onChange : null}
              disabled={!isActive}
            />
            <span className={toggleClasses}></span>
          </label>
        </div>
      </div>
    </div>
  );
}

function SLAMControlPanel({ slamMode, onLocalizationMode, onMappingMode, onResetMap, saveMap }) {
  const isActive = (slamMode !== config.slam_mode.IDLE) && (slamMode !== config.slam_mode.INVALID);
  return (
    <>
      <ToggleSelect
        label={"Localization Mode"}
        explain={"Toggles localization mode and displays map."}
        checked={isActive}
        onChange={onLocalizationMode}
      />
      {isActive &&
        <div className="subpanel">
          <ToggleSelect
            label={"Mapping Mode"}
            checked={slamMode === config.slam_mode.FULL_SLAM}
            explain={"Toggles mapping mode on the robot."}
            onChange={onMappingMode}
            small={true}
          />
          <div className="button-wrapper-col">
            <button
              className={"button" + (slamMode !== config.slam_mode.FULL_SLAM ? " inactive" : "")}
              onClick={onResetMap}
            >
              Reset Map
            </button>
            <button
              className="button"
              onClick={saveMap}
            >
              Download Map
            </button>
          </div>
        </div>
      }
    </>
  );
}

function MBotSceneWrapper({ mbot, scene, connected, slamMode, robotDisplay, laserDisplay, particleDisplay, mapDisplay,
                            poseAvailable, laserAvailable, mapAvailable, particlesAvailable, slamModeAvailable,
                            setClickedCell, setRobotPose, setRobotCell}) {
  // Ref for the canvas.
  const canvasWrapperRef = useRef(null);

  // Click callback when the user clicks on the scene.
  const handleCanvasClick = useCallback((pos) => {
    if (!scene.current.loaded) return;
    if (pos.length === 0 || !scene.current.isMapLoaded()) {
      // If the map is not loaded or an empty cell is passed, clear.
      setClickedCell([]);
      return;
    }

    const clickedCell = [...scene.current.pixelsToCell(pos[0], pos[1]),
                         ...scene.current.pixelsToPos(pos[0], pos[1])];
    setClickedCell(clickedCell);
  }, [setClickedCell]);

  // Initialization of the scene.
  useEffect(() => {
    scene.current.init().then(() => {
      scene.current.createScene(canvasWrapperRef.current);
      scene.current.clickCallback = handleCanvasClick;
    }).catch((error) => {
      console.warn(error);
    });

    return () => {
      // Clean up.
    }
  }, [canvasWrapperRef, handleCanvasClick]);

  // Effect to manage subscribing to the pose.
  useEffect(() => {
    if (scene.current.loaded) scene.current.toggleRobotView(robotDisplay);

    if (connected && robotDisplay && poseAvailable) {
      mbot.subscribe(config.POSE_CHANNEL, (msg) => {
        // Sets the robot position
        setRobotPose({x: msg.data.x, y: msg.data.y, theta: msg.data.theta});
        if (!scene.current.loaded) return;
        scene.current.updateRobot(msg.data.x, msg.data.y, msg.data.theta);
        if (scene.current.isMapLoaded()) {
          const robotCell = scene.current.posToCell(msg.data.x, msg.data.y);
          setRobotCell(robotCell);
        }
      }).catch((error) => {
        console.warn('Subscription failed for channel', config.POSE_CHANNEL, error);
      });
    }

    // Return the cleanup function which stops the rerender.
    return () => {
      if (connected) mbot.unsubscribe(config.POSE_CHANNEL).catch((err) => console.warn(err));
    }
  }, [connected, robotDisplay, poseAvailable, setRobotPose, setRobotCell]);

  // Effect to manage subscribing to the Lidar.
  useEffect(() => {
    if (connected && laserDisplay && laserAvailable) {
      mbot.subscribe(config.LIDAR_CHANNEL, (msg) => {
        if (!scene.current.loaded) return;
        scene.current.drawLasers(msg.data.ranges, msg.data.thetas);
      }).catch((error) => {
        console.warn('Subscription failed for channel', config.LIDAR_CHANNEL, error);
      });
    }
    else {
      if (scene.current.loaded) scene.current.clearLasers();
    }

    // Return the cleanup function which stops the rerender.
    return () => {
      if (connected) mbot.unsubscribe(config.LIDAR_CHANNEL).catch((err) => console.warn(err));
    }
  }, [connected, laserAvailable, laserDisplay]);

  // Effect to manage subscribing to the SLAM particles.
  useEffect(() => {
    if (connected && particleDisplay && particlesAvailable) {
      mbot.subscribe(config.PARTICLE_CHANNEL, (msg) => {
        if (!scene.current.loaded) return;
        // Extract the particles into points.
        const particleList = msg.data.particles;
        const points = particleList.map(item => [item.pose.x, item.pose.y]);
        // Draw the particles.
        scene.current.drawParticles(points);
      }).catch((error) => {
        console.warn('Subscription failed for channel', config.PARTICLE_CHANNEL, error);
      });
    }
    else {
      if (scene.current.loaded) scene.current.clearParticles();
    }

    // Return the cleanup function which stops the rerender.
    return () => {
      if (connected) mbot.unsubscribe(config.PARTICLE_CHANNEL).catch((err) => console.warn(err));
    }
  }, [connected, particleDisplay, particlesAvailable]);

  // Effect to manage subscribing to the path.
  useEffect(() => {
    if (connected) {
      mbot.subscribe(config.PATH_CHANNEL, (msg) => {
        if (!scene.current.loaded) return;
        const pathPoints = msg.data.path;
        if (pathPoints.length === 0) {
          scene.current.clearPath();  // If path length is zero, clear and return.
        }
        else {
          // Extract coordinates of the path and draw.
          const points = pathPoints.map(item => [item.x, item.y]);
          scene.current.drawPath(points);
        }
      }).catch((error) => {
        console.warn('Subscription failed for channel', config.PATH_CHANNEL, error);
      });
    }

    // Return the cleanup function which stops the rerender.
    return () => {
      if (connected) mbot.unsubscribe(config.PATH_CHANNEL).catch((err) => console.warn(err));
    }
  }, [connected]);

  // Effect to request the SLAM map.
  useEffect(() => {
    let timerId = null;
    let mapRequestCount = 0;

    async function requestSLAMMap() {
      try {
        const data = await mbot.readMap();
        const headerData = {
          width: data.width,
          height: data.height,
          metersPerCell: data.meters_per_cell,
          origin: data.origin
        };

        if (scene.current.loaded) {
          scene.current.setMapHeaderData(data.width, data.height, data.meters_per_cell, data.origin);
          scene.current.updateCells(data.cells);
        }

        mapRequestCount = 0;  // If the map was retrieved, reset the fail count.
        return headerData;
      } catch (error) {
        mapRequestCount++;  // Keep track of how many failed map requests we have made.
        return null;
      }
    }

    if (scene.current.loaded) {
      scene.current.clear();  // Clear the scene on change.
    }

    // If we are making a map OR there is no SLAM mode but the user has
    // requested to visualize the map, request the map at a regular interval.
    // If the request fails, after a timeout period the request will stop.
    if ((!slamModeAvailable && mapDisplay) ||
        slamMode === config.slam_mode.FULL_SLAM ||
        slamMode === config.slam_mode.MAPPING_ONLY){
      // Check for map once right away.
      requestSLAMMap();
      // Check for map intermittently.
      timerId = setInterval(() => { requestSLAMMap().then((val) => {
        if (mapRequestCount > config.STALE_MAP_COUNT) {
          // Timeout condition. Give up full SLAM mode and reset.
          console.warn("No map available! Resetting SLAM to IDLE.");
          if (slamModeAvailable) mbot.resetSLAM(config.slam_mode.IDLE);
          clearInterval(timerId);
        }
      }); }, config.MAP_UPDATE_PERIOD);
    }
    else if (slamMode === config.slam_mode.LOCALIZATION_ONLY) {
      // Try requesting the map only once if we are in localization mode.
      requestSLAMMap().then((val) => {
        if (!val) {
          // If we didn't get a SLAM map, keep asking for one until we get one or we timeout.
          timerId = setInterval(() => {
            requestSLAMMap().then((val) => {
              if (val) clearInterval(timerId); // If we get a map, stop requesting.
              if (mapRequestCount > config.STALE_MAP_COUNT) {
                // Timeout condition. Give up localization mode and reset to IDLE.
                console.warn("No map available! Resetting SLAM to IDLE.");
                mbot.resetSLAM(config.slam_mode.IDLE);
                clearInterval(timerId);
              }
            });
          }, config.MAP_UPDATE_PERIOD);
        }
      });
    }

    // On quit, stop requesting.
    return () => {
      if (timerId) clearInterval(timerId);
    };
  }, [slamMode, slamModeAvailable, mapDisplay]);

  return (
    <div id="canvas-container" ref={canvasWrapperRef}>
    </div>
  );
}

export default function MBotApp({ mbot }) {
  const scene = useRef(new MBotScene());
  const [hostname, setHostname] = useState("mbot-???");
  const [connected, setConnected] = useState(false);
  // Toggle selectors.
  const [robotDisplay, setRobotDisplay] = useState(true);
  const [laserDisplay, setLaserDisplay] = useState(false);
  const [particleDisplay, setParticleDisplay] = useState(false);
  const [mapDisplay, setMapDisplay] = useState(false);
  const [drivingMode, setDrivingMode] = useState(false);
  // Channels to subscribe to.
  const [poseAvailable, setPoseAvailable] = useState(false);
  const [laserAvailable, setLaserAvailable] = useState(false);
  const [mapAvailable, setMapAvailable] = useState(false);
  const [particlesAvailable, setParticlesAvailable] = useState(false);
  const [slamModeAvailable, setSlamModeAvailable] = useState(false);
  // Robot parameters.
  const [robotPose, setRobotPose] = useState({x: 0, y: 0, theta: 0});
  const [robotCell, setRobotCell] = useState([0, 0]);
  // Visualization elements.
  const [clickedCell, setClickedCell] = useState([]);
  // Mapping parameters.
  const [slamMode, setSlamMode] = useState(config.slam_mode.INVALID);
  // Saves the SLAM mode globally to avoid having to condition the status effect on the slamMode state.
  const latestSlamMode = useRef(config.slam_mode.INVALID);

  // A heartbeat effect that checks if the MBot Bridge backend is connected and
  // updates which channels we care about are available.
  useEffect(() => {
    let timerId = null;

    const chMaps = [
      {state: poseAvailable, ch: config.POSE_CHANNEL, setter: setPoseAvailable},
      {state: laserAvailable, ch: config.LIDAR_CHANNEL, setter: setLaserAvailable},
      {state: mapAvailable, ch: config.SLAM_MAP_CHANNEL, setter: setMapAvailable},
      {state: particlesAvailable, ch: config.PARTICLE_CHANNEL, setter: setParticlesAvailable},
      {state: slamModeAvailable, ch: config.SLAM_MODE_CHANNEL, setter: setSlamModeAvailable},
    ]

    function checkChannels() {
      mbot.readChannels().then((chs) => {
        const chsList = chs.map((ch) => ch.channel);
        if (!connected) setConnected(true);
        // Check if any of the channels we are looking for have either appeared or disappeared.
        for (const ele of chMaps) {
          const hasData = chsList.includes(ele.ch);
          if (hasData != ele.state) ele.setter(hasData);
        }
      }).catch((err) => {
        if (connected) setConnected(false);
      });
    }

    // Check if connected once right away.
    checkChannels();

    // Check for connection intermittently.
    timerId = setInterval(() => { checkChannels(); }, config.CONNECT_PERIOD);

    // Return the cleanup function which stops the rerender.
    return () => {
      if (timerId) clearInterval(timerId);
    };
  }, [connected, setConnected,
      poseAvailable, setPoseAvailable,
      laserAvailable, setLaserAvailable,
      mapAvailable, setMapAvailable,
      particlesAvailable, setParticlesAvailable,
      slamModeAvailable, setSlamModeAvailable]);

  // Effect to request the MBot hostname on first mounting component.
  useEffect(() => {
    // Read the hostname.
    if (connected) {
      mbot.readHostname().then((name) => {
        setHostname(name);
      }).catch((err) => {
        console.warn("Could not get hostname:", err);
      });
    }
  }, [connected, setHostname]);

  // Effect to manage SLAM mode.
  useEffect(() => {
    if (!slamModeAvailable) return;

    mbot.subscribe(config.SLAM_MODE_CHANNEL, (msg) => {
      const data = msg.data;
      // Only update if the mode has changed.
      if (data.slam_mode !== latestSlamMode.current) {
        setSlamMode(data.slam_mode);
        latestSlamMode.current = data.slam_mode;
      }
    }).then().catch((error) => {
      console.warn('Subscription failed for channel', config.SLAM_MODE_CHANNEL, error);
    });

    // Return the cleanup function which stops the subscription.
    return () => {
      mbot.unsubscribe(config.SLAM_MODE_CHANNEL).catch((err) => console.warn(err));
    }
  }, [slamModeAvailable, latestSlamMode, setSlamMode]);

  // Callbacks.
  const onLocalizationMode = useCallback(() => {
    if (slamMode === config.slam_mode.IDLE) {
      // State is idle. Change to localization only.
      mbot.resetSLAM(config.slam_mode.LOCALIZATION_ONLY, false);
      setSlamMode(config.slam_mode.LOCALIZATION_ONLY);
    }
    else if (slamModeAvailable) {
      // We are in some other state. Turn back to idle.
      mbot.resetSLAM(config.slam_mode.IDLE);
      setSlamMode(config.slam_mode.IDLE);
    }
  }, [slamModeAvailable, slamMode, setSlamMode]);

  const onMappingMode = useCallback(() => {
    if (slamMode === config.slam_mode.FULL_SLAM) {
      // If we're in full slam, we need to reset the robot to localization only mode.
      mbot.resetSLAM(config.slam_mode.LOCALIZATION_ONLY, true);
      setSlamMode(config.slam_mode.LOCALIZATION_ONLY);
    }
    else if (slamMode === config.slam_mode.LOCALIZATION_ONLY) {
      // If we are not mapping, we need to tell the robot to start mapping.
      if (!confirm("This will overwrite the current map. Are you sure?")) return;

      mbot.resetSLAM(config.slam_mode.FULL_SLAM, false);
      setSlamMode(config.slam_mode.FULL_SLAM);
    }
  }, [slamMode, setSlamMode]);

  const onResetMap = useCallback(() => {
    if (slamMode === config.slam_mode.FULL_SLAM) {
      // Get user confirmation that the map should be cleared.
      if (!confirm("This will clear the current map. Are you sure?")) return;
      // Reset in full SLAM mode.
      mbot.resetSLAM(config.slam_mode.FULL_SLAM, false);
    }
  }, [slamMode]);

  const saveMap = useCallback(() => {
    if (!scene.current.loaded) return;
    const mapData = scene.current.getMapData();

    if (mapData === null) {
      console.log("Error saving map: Invalid map data");
      return;
    }

    downloadMapFile(mapData);
  }, []);

  return (
    <div id="wrapper">
      <div id="main">
        <MBotSceneWrapper mbot={mbot} scene={scene} connected={connected}
                          robotDisplay={robotDisplay}
                          laserDisplay={laserDisplay}
                          particleDisplay={particleDisplay}
                          mapDisplay={mapDisplay}
                          slamMode={slamMode}
                          poseAvailable={poseAvailable}
                          laserAvailable={laserAvailable}
                          mapAvailable={mapAvailable}
                          particlesAvailable={particlesAvailable}
                          slamModeAvailable={slamModeAvailable}
                          setClickedCell={setClickedCell}
                          setRobotPose={setRobotPose}
                          setRobotCell={setRobotCell} />
      </div>

      <div id="sidenav">
        <div id="toggle-nav" onClick={() => {}}><FontAwesomeIcon icon={faBars} /></div>
        <div className="inner">
          <div className="title">
            {hostname.toUpperCase()}
          </div>

          <div className="status-wrapper">
            <ConnectionStatus status={connected}/>
            <StatusMessage robotCell={robotCell} robotPose={robotPose}
                           clickedCell={clickedCell} />
          </div>

          <div className="row">
            {/* Only show the SLAM control panel if we have received a SLAM status message. */}
            {slamModeAvailable &&
                <SLAMControlPanel slamMode={slamMode}
                                  onLocalizationMode={() => onLocalizationMode()}
                                  onMappingMode={() => onMappingMode()}
                                  onResetMap={() => onResetMap()}
                                  saveMap={() => saveMap()} />
              }

              {/* If there is no SLAM mode, provide the option to display the map, if available. */}
              {!slamModeAvailable &&
               <ToggleSelect label={"Draw Map"} checked={mapDisplay} isActive={mapAvailable}
                             explain={"Displays the SLAM map."}
                             onChange={ () => { setMapDisplay(!mapDisplay); } }/>
              }

              { /* Checkboxes for map visualization. */}
              <ToggleSelect label={"Draw Robot"} checked={robotDisplay}
                            explain={"Displays the robot on the map."}
                            onChange={ () => { setRobotDisplay(!robotDisplay); } }/>

              <ToggleSelect label={"Draw Particles"} checked={particleDisplay} isActive={particlesAvailable}
                            explain={"Shows all the positions the robot thinks it might be at."}
                            onChange={ () => { setParticleDisplay(!particleDisplay); } }/>

              <ToggleSelect label={"Draw Lasers"} checked={laserDisplay} isActive={laserAvailable}
                            explain={"Displays the Lidar rays."}
                            onChange={ () => { setLaserDisplay(!laserDisplay); } }/>

              { /* Drive mode and control panel. */}
              <ToggleSelect label={"Drive Mode"} checked={drivingMode}
                            explain={"To drive the robot with your keyboard, use A,D for left & right, " +
                                      "W,S for forward & backward, and Q,E to rotate. " +
                                      "Or, use the joystick and turn buttons in the drive panel."}
                            onChange={ () => { setDrivingMode(!drivingMode); } }/>
              {drivingMode &&
                <DriveControlPanel mbot={mbot} drivingMode={drivingMode} />
              }
          </div>
        </div>
      </div>
    </div>
  );
}
