import { Container, Graphics, GraphicsContext, Application, Assets, Sprite, Point } from 'pixi.js';

import config from "./config.js";

 function colourStringToRGB(colour_str) {
  var rgb = [parseInt(colour_str.substring(1, 3), 16),
             parseInt(colour_str.substring(3, 5), 16),
             parseInt(colour_str.substring(5, 7), 16)];
  return rgb;
}

function getColor(prob, colour_low, colour_high) {
  // Takes a probability (number from 0 to 1) and converts it into a color code
  var colour_low_a = colourStringToRGB(colour_low);
  var colour_high_a = colourStringToRGB(colour_high);

  var hex = function(x) {
    x = x.toString(16);
    return (x.length == 1) ? '0' + x : x;
  };

  var r = Math.ceil(colour_high_a[0] * prob + colour_low_a[0] * (1 - prob));
  var g = Math.ceil(colour_high_a[1] * prob + colour_low_a[1] * (1 - prob));
  var b = Math.ceil(colour_high_a[2] * prob + colour_low_a[2] * (1 - prob));

  var color = hex(r) + hex(g) + hex(b);
  return "#" + color;
}


class OccupancyGrid {
  constructor(viewSize, colors = [config.MAP_COLOUR_LOW, config.MAP_COLOUR_HIGH]) {
    // Map data.
    this.width = 0;           // Map width in cells.
    this.height = 0;         // Map height in cells.
    this.metersPerCell = 0.05;
    this.origin = [0, 0];
    this.pixWidth = viewSize[0];   // Map width in pixels.
    this.pixHeight = viewSize[1];  // Map height in pixels.
    this.pixPerCell = 5;
    this.pixelsPerMeter = this.pixPerCell / this.metersPerCell;
    this.colorRange = colors;  // Colours to interpolate between for display.

    this.mapCells = [];  // Current cell values.
    this.gridMap = {};

    this.gridContainer = null;  // Container for holding the grid render objects.
    this.cellContext = null;    // Shared context for drawing the cells.
  }

  cellToPixels(r, c) {
    let v = ((this.height - r - 1) * this.pixPerCell);
    let u = (c * this.pixPerCell);
    return [u, v];
  }

  pixelsToCell(u, v) {
    let row = Math.floor((this.pixHeight - v) / this.pixPerCell);
    let col = Math.floor(u / this.pixPerCell);
    return [row, col];
  }

  posToCell(x, y) {
    let i = Math.floor((x - this.origin[0]) / this.metersPerCell);
    let j = Math.floor((y - this.origin[1]) / this.metersPerCell);
    return [j, i];
  }

  cellToIdx(r, c) {
    return c + r * this.width;
  }

  idxToCell(idx) {
    let r = Math.floor(idx / this.width);
    let c = idx % this.width;
    return [r, c];
  }

  setMapData(width, height, m_per_cell = 0.05, origin = [0, 0]) {
    // If this data is the same, no need to recompute or clear.
    if (width === this.width && height === this.height &&
        m_per_cell === this.metersPerCell &&
        origin[0] === this.origin[0] && origin[1] === this.origin[1]) return;

    this.width = width;
    this.height = height;
    this.metersPerCell = m_per_cell;
    this.origin = origin;
    this.pixPerCell = this.pixWidth / this.width;
    this.pixelsPerMeter = this.pixWidth / this.width / m_per_cell;

    this.clear();

    // Make a new cell context of the right size.
    if (this.cellContext) this.cellContext.destroy();
    this.cellContext = new GraphicsContext()
      .rect(0, 0, this.pixPerCell, this.pixPerCell)
      .fill(0xffffff);
  }

  getMapData() {
    if (this.mapCells.length !== this.width * this.height) {
      return;
    }

    let mapData = {"cells": this.mapCells,
                   "width": this.width,
                   "height": this.height,
                   "origin": this.origin,
                   "metersPerCell": this.metersPerCell};
    return mapData;
  }

  init() {
    if (this.gridContainer) this.gridContainer.destroy();

    // This container will hold all the grid cells.
    this.gridContainer = new Container();
    // Make the background all grey so we can ignore any uncertain cells.
    let bkgd = new Graphics();
    bkgd.rect(0, 0, this.pixWidth, this.pixHeight)
        .fill(getColor(0.5, this.colorRange[0], this.colorRange[1]));
    this.gridContainer.addChild(bkgd);

    if (this.cellContext) this.cellContext.destroy();
    // Create a cell context to be used to draw cells later.
    this.cellContext = new GraphicsContext()
      .rect(0, 0, this.pixPerCell, this.pixPerCell)
      .fill(0xffffff);

    this.gridMap = {};
    this.mapCells = [];

    return this.gridContainer;
  }

  clear() {
    if (this.gridContainer) {
      // Clear all the current grid cells. Skip the first since it's the background.
      for (let i = 1; i < this.gridContainer.children.length; i++) {
        this.gridContainer.children[i].destroy();
      }
      this.gridContainer.children.splice(1);  // Remove cells from the list. Again, keep background.
    }
    this.gridMap = {};

    // Reset the cells.
    this.mapCells = [];
  }

  updateCells(cells) {
    if (cells.length !== this.width * this.height) {
      console.warn("Error. Cannot render canvas: " + String(cells.length) + " != " + String(this.width*this.height));
      return;
    }

    if (cells.length !== this.mapCells.length) {
      this.clear();
    }

    for (let c = 0; c < this.width; c++) {
      for (let r = 0; r < this.height; r++) {
        let idx = this.cellToIdx(r, c);
        if (cells[idx] != this.mapCells[idx]){
          let prob = (cells[idx] + 127.) / 255.;
          let color = getColor(prob, this.colorRange[0], this.colorRange[1]);
          let pos = this.cellToPixels(r, c);

          if (this.gridMap[idx]) {
            // If we already have a grid cell at this location, just update its color.
            this.gridContainer.children[this.gridMap[idx]].tint = parseInt(color.slice(1), 16);
          }
          else {
            // Skip any cells that already the colour of the background.
            if (cells[idx] == 0) continue;
            // If there was not a grid cell here, create a new one.
            let cell = new Graphics(this.cellContext);
            cell.x = pos[0];
            cell.y = pos[1];
            cell.tint = parseInt(color.slice(1), 16);
            this.gridContainer.addChild(cell);
            // Save the order of this cell in the children.
            this.gridMap[idx] = this.gridContainer.children.length - 1;
          }
        }
      }
    }

    this.mapCells = cells;
  }

}


class MBotScene {
  constructor() {
    this.app = null;

    this.robotState = {x: 0, y: 0, theta: 0};
    // Map data.
    const DEFAULT_WIDTH = 100;  // Default number of cells if no map is provided.
    this.metersPerCell = 0.05;  // Default meters per cell.
    const origin = -this.metersPerCell * DEFAULT_WIDTH / 2;  // By default, origin is in the center.
    this.origin = [origin, origin];
    this.pixWidth = config.CANVAS_DISPLAY_WIDTH;
    this.pixHeight = config.CANVAS_DISPLAY_HEIGHT;
    this.pixPerCell = this.pixWidth / DEFAULT_WIDTH;
    this.pixelsPerMeter = this.pixPerCell / this.metersPerCell;

    this.occupancyGrid = new OccupancyGrid([this.pixWidth, this.pixHeight]);
    this.occupancyGrid.setMapData(DEFAULT_WIDTH, DEFAULT_WIDTH, this.metersPerCell, this.origin);

    this.dragStart = null;
    this.clickStart = null;
    this.clickCallback = (pos) => {};

    this.stopped = false;
    this.loaded = false;
  }

  async init() {
    return new Promise(async (resolve, reject) => {

      if (this.stopped) {
        this.loaded = false;
        reject(new Error("Initialization aborted because destroy was called."));
        return;
      }

      if (this.loaded || this.app) {
        reject("App is already initialized or initialization was already called.");
      }

      this.app = new Application();
      this.loaded = false;

      try {
        await this.app.init({resizeTo: window, backgroundColor: 0xc9d1d9 });

        const imgUrl = new URL('./images/mbot.png', import.meta.url).href;
        this.robotImage = await Assets.load(imgUrl);

        // Check one last time before considering initialization complete
        if (this.stopped) {
          this.app.destroy(true, true);
          reject("Initialization aborted because destroy was called.");
        } else {
          resolve();
        }
      } catch (error) {
        reject(error); // Ensure any errors are caught and the promise is rejected
      }
    });
  }

  destroy() {
    if (this.app && this.loaded) {
      this.app.destroy(true, true);
      this.loaded = false;
    }
    this.stopped = true;
  }

  createScene(ele) {
    this.app.resizeTo = ele;
    ele.appendChild(this.app.canvas);

    this.sceneContainer = new Container();
    this.app.stage.addChild(this.sceneContainer);

    // Initialize the occupancy grid and add its container to the scene.
    let gridContainer = this.occupancyGrid.init();
    this.sceneContainer.addChild(gridContainer);

    // Empty graphics to draw particles.
    this.particlesGraphics = new Graphics();
    this.sceneContainer.addChild(this.particlesGraphics);

    // Empty graphics to draw path.
    this.pathGraphics = new Graphics();
    this.sceneContainer.addChild(this.pathGraphics);

    // Empty graphics to draw clicked cell.
    this.clickedCellGraphics = new Graphics();
    this.sceneContainer.addChild(this.clickedCellGraphics);

    // Robot Container.
    this.robotContainer = new Container();

    // Empty graphics to draw the lasers, in the robot frame.
    this.laserGraphics = new Graphics();
    this.robotContainer.addChild(this.laserGraphics);

    this.robot = new Sprite(this.robotImage);
    this.robot.anchor.set(0.5);
    // Move the sprite to the center of the screen
    this.robotContainer.x = this.pixWidth / 2;
    this.robotContainer.y = this.pixHeight / 2;
    this.robot.x = 0;
    this.robot.y = 0;
    this.robot.width = config.ROBOT_SIZE * this.pixelsPerMeter;
    this.robot.height = config.ROBOT_SIZE * this.pixelsPerMeter;
    this.robotContainer.addChild(this.robot);

    this.sceneContainer.addChild(this.robotContainer);

    // Interaction code for panning
    this.dragStart = null;
    this.sceneContainer.interactive = true;
    this.sceneContainer.on('pointerdown', (event) => {
        this.dragStart = event.data.getLocalPosition(this.sceneContainer.parent);
        this.clickStart = event.data.getLocalPosition(this.sceneContainer.parent);
    });

    this.sceneContainer.on('pointerup', (event) => {
      let upPos = event.data.getLocalPosition(this.sceneContainer.parent);
      if (this.clickStart) {
        // Detect a click if the pointer was raised close to where it started.
        if (Math.abs(this.clickStart.x - upPos.x) < 1 &&
            Math.abs(this.clickStart.y - upPos.y) < 1) {
          // Convert the clicked point into the scene frame in order to account for scaling and zooming.
          let localPos = this.sceneContainer.toLocal(new Point(this.clickStart.x, this.clickStart.y));
          // Draw the cell.
          this.drawClickedCell(localPos.x, localPos.y);
          // Call any user-defined click callback.
          this.clickCallback([localPos.x, localPos.y]);
        }
      }
      this.dragStart = null;
      this.clickStart = null;
      this.constrainSceneContainer();
    });

    this.sceneContainer.on('pointerupoutside', (event) => {
      this.dragStart = null;
      this.clickStart = null;
      this.constrainSceneContainer();
    });

    this.sceneContainer.on('pointermove', (event) => {
        if (this.dragStart) {
            const dragEnd = event.data.getLocalPosition(this.sceneContainer.parent);
            const dragNew = {
                x: dragEnd.x - this.dragStart.x,
                y: dragEnd.y - this.dragStart.y,
            };

            this.sceneContainer.x += dragNew.x;
            this.sceneContainer.y += dragNew.y;
            this.dragStart = dragEnd;
        }
    });

    // Interaction code for zooming
    this.app.canvas.addEventListener('wheel', (event) => { this.zoomHandler(event); });

    this.loaded = true;
  }

  zoomHandler(event) {
    event.preventDefault();
    const scaleFactor = 1.1;
    let globalPos = this.sceneContainer.toLocal(new Point(event.x, event.y));
    // The farthest out you can scale (assuming that the grid is square).
    let minScale = Math.min(this.app.canvas.width, this.app.canvas.height) / (this.pixWidth);
    const direction = event.deltaY > 0 ? -1 : 1;  // Negative if scrolling up, positive if down
    const scale = Math.pow(scaleFactor, direction);
    let scaleX = this.sceneContainer.scale.x * scale;
    let scaleY = this.sceneContainer.scale.y * scale;

    if (scaleX > minScale && scaleY > minScale) {
        // Zoom
        this.sceneContainer.scale.set(scaleX); // x = scaleX;
        this.sceneContainer.pivot.x = globalPos.x;
        this.sceneContainer.pivot.y = globalPos.y;
        this.sceneContainer.position.set(event.x, event.y);
    }
    else {
      // Don't zoom out more than the size of the screen.
        this.sceneContainer.scale.set(minScale);
        this.sceneContainer.position.set(0, 0);
        this.sceneContainer.pivot.x = 0;
        this.sceneContainer.pivot.y = 0;
    }

    this.constrainSceneContainer();
  }

  constrainSceneContainer() {
    // Don't let the scene container go out of the view more than it needs to.
    let pt0 = this.sceneContainer.toLocal(new Point(0, 0));
    let pt1 = this.sceneContainer.toLocal(new Point(this.app.canvas.width, this.app.canvas.height));
    if (pt0.x < 0) {
      this.sceneContainer.pivot.x = 0;
      this.sceneContainer.x = 0;
    }
    else if (pt1.x > this.pixWidth) {
      let new_x = this.sceneContainer.x + this.sceneContainer.scale.x * (pt1.x - this.pixWidth);
      if (this.pixWidth * this.sceneContainer.scale.x < this.app.canvas.width) {
        // If the scene is smaller than the screen, don't snap to the far side.
        this.sceneContainer.pivot.x = 0;
        new_x = Math.min(new_x, 0);
      }
      this.sceneContainer.x = new_x;
    }
    if (pt0.y < 0) {
      this.sceneContainer.pivot.y = 0;
      this.sceneContainer.y = 0;
    }
    else if (pt1.y > this.pixHeight) {
      let new_y = this.sceneContainer.y + this.sceneContainer.scale.x * (pt1.y - this.pixHeight);
      if (this.pixHeight * this.sceneContainer.scale.x < this.app.canvas.height) {
        // If the scene is smaller than the screen, don't snap to the bottom.
        this.sceneContainer.pivot.y = 0;
        new_y = Math.min(new_y, 0);
      }
      this.sceneContainer.y = new_y;
    }
  }

  setMapHeaderData(width, height, m_per_cell = 0.05, origin = [0, 0]) {
    this.occupancyGrid.setMapData(width, height, m_per_cell, origin);
    this.metersPerCell = m_per_cell;
    this.pixPerCell = this.pixWidth / width;
    this.pixelsPerMeter = this.pixWidth / width / m_per_cell;
    this.origin = origin;

    this.robot.width = config.ROBOT_SIZE * this.pixelsPerMeter;
    this.robot.height = config.ROBOT_SIZE * this.pixelsPerMeter;
  }

  getMapData() {
    return this.occupancyGrid.getMapData();
  }

  isMapLoaded() {
    return this.occupancyGrid.mapCells.length > 0;
  }

  posToPixels(x, y) {
    let u = (x - this.origin[0]) * this.pixelsPerMeter;
    let v = this.pixHeight - (y - this.origin[1]) * this.pixelsPerMeter;
    return [u, v];
  }

  pixelsToPos(u, v){
    let x = (u / this.pixelsPerMeter) + this.origin[0];
    let y = (this.pixHeight - v) / this.pixelsPerMeter + this.origin[1];
    return [x, y];
  }

  cellToPixels(r, c) {
    return this.occupancyGrid.cellToPixels(r, c);
  }

  pixelsToCell(u, v) {
    return this.occupancyGrid.pixelsToCell(u, v);
  }

  posToCell(x, y) {
    return this.occupancyGrid.posToCell(x, y);
  }

  clear() {
    // Clear the occupancy grid.
    this.occupancyGrid.clear();

    // Clear all saved graphics.
    this.pathGraphics.clear();
    this.particlesGraphics.clear();
    this.clickedCellGraphics.clear();
  }

  updateRobot(x, y, theta = 0) {
    let pix = this.posToPixels(x, y);
    this.robotContainer.x = pix[0];
    this.robotContainer.y = pix[1];
    this.robotContainer.rotation = -theta;

    this.robotState.x = x;
    this.robotState.y = y;
    this.robotState.theta = theta;
  }

  toggleRobotView(visible) {
    this.robot.visible = visible;
  }

  updateCells(cells) {
    // If this map is fully new, clear any other components on the screen.
    if (cells.length !== this.occupancyGrid.mapCells.length) {
      this.clearPath();
      this.clearClickedCell();
    }

    this.occupancyGrid.updateCells(cells);
  }

  drawLasers(ranges, thetas, color = "green", line_width = 0.5) {
    this.laserGraphics.clear();
    this.laserGraphics.beginPath();
    for (var i = 0; i < ranges.length; i++) {
      this.laserGraphics.moveTo(0, 0);
      let rayX = ranges[i] * Math.cos(thetas[i]) * this.pixelsPerMeter;
      let rayY = -ranges[i] * Math.sin(thetas[i]) * this.pixelsPerMeter;
      this.laserGraphics.lineTo(rayX, rayY);
    }

    this.laserGraphics.stroke({ width: line_width, color: color });
  }

  clearLasers() {
    if (this.laserGraphics) this.laserGraphics.clear();
  }

  drawPath(path, color = "rgb(255, 25, 25)", line_width = 0.5) {
    this.pathGraphics.clear();
    if (path.length === 0) return;  // Don't draw if the path is empty.

    this.pathGraphics.beginPath();
    let current = this.posToPixels(path[0][0], path[0][1]);
    this.pathGraphics.moveTo(current[0], current[1]);
    for(let i = 1; i < path.length; i++) {
      // Draws a line between the points
      current = this.posToPixels(path[i][0], path[i][1]);
      this.pathGraphics.lineTo(current[0], current[1]);
    }
    this.pathGraphics.stroke({ width: line_width, color: color });
  }

  clearPath() {
    this.pathGraphics.clear();
  }

  drawParticles(particles, color = 0xff0000, size = 1){
    this.particlesGraphics.clear();
    for (let i = 0; i < particles.length; i++) {
      let pt = this.posToPixels(particles[i][0], particles[i][1])
      this.particlesGraphics.circle(pt[0], pt[1], size * this.pixPerCell).fill(color)
    }
  }

  clearParticles() {
    this.particlesGraphics.clear();
  }

  drawClickedCell(u, v) {
    const cell = this.pixelsToCell(u, v);
    let pos = this.cellToPixels(cell[0], cell[1]);

    this.clickedCellGraphics.clear()
    this.clickedCellGraphics.rect(pos[0], pos[1], this.pixPerCell, this.pixPerCell).fill(0xff8300);
  }

  clearClickedCell() {
    this.clickedCellGraphics.clear();
    this.clickCallback([]);
  }
}

export { colourStringToRGB, getColor, MBotScene };
