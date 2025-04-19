/*******************
 * MAP HELPERS
 *******************/

function parseMapFromSocket(data) {
  var map = {};
  var datacells = [];
  var count = 0;

  for (let line of data.split(" ")) {
    if (line!="\n") {
      datacells.push(parseFloat(line.replace("[", "").replace("]", "").replace('"', '').replace("\\n", "")));
      count++;
    }
  }

  map.origin = [parseFloat(datacells.shift()), parseFloat(datacells.shift())];
  map.width = parseFloat(datacells.shift());
  map.height = parseFloat(datacells.shift());
  map.meters_per_cell = parseFloat(datacells.shift());
  map.num_cells = map.width * map.height;

  map.cells = normalizeList(datacells);

  if (map.cells.length!=map.width*map.height) {

  }

  return map;
}

function parseMapFromLcm(msg){
  var map = {};
  map.origin = msg["origin"];
  map.width = msg["width"];
  map.height = msg["height"];
  map.meters_per_cell = msg["meters_per_cell"];
  map.num_cells = msg["num_cells"];
  map.cells = msg["cells"];
  map.slam_mode = msg["slam_mode"];
  map.slam_map_location = msg["slam_map_location"];
  return map;
}


function mapDataToString(data){
  // Add the header to the string.
  let header = [...data.origin, data.width, data.height, data.metersPerCell];
  let res = header.join(" ") + "\n";

  // Add cell data.
  for (let i = 0; i < data.height; i++) {
    res += data.cells.slice(data.width * i, data.width * (i + 1)).join(" ") + "\n";
  }
  return res;
}


function normalizeList(list) {
  if (list.length < 1) return list;
  if (list.length === 1) return [1];

  var normalizedList = [...list]

  // Find min and max values.
  // Note: Using JavaScript's Math.min(...) and Math.min(...) causes issues if
  // the array is too big to unpack.
  var min_val = normalizedList[0];
  var max_val = normalizedList[0];

  for (var i = 1; i < normalizedList.length; i++) {
    min_val =  normalizedList[i] < min_val ? normalizedList[i] : min_val;
    max_val =  normalizedList[i] > max_val ? normalizedList[i] : max_val;
  }

  // Normalize the values.
  for (var i = 0; i < normalizedList.length; i++) {
    normalizedList[i] = (normalizedList[i] - min_val) / (max_val - min_val);
  }

  return normalizedList;
}

function downloadMapFile(mapData){
  var dataStr = "data:text/plain;charset=utf-8," + encodeURIComponent(mapDataToString(mapData));
  var downloadAnchorNode = document.createElement('a');
  downloadAnchorNode.setAttribute("href", dataStr);
  downloadAnchorNode.setAttribute("download", "current.map");
  document.body.appendChild(downloadAnchorNode); // required for firefox
  downloadAnchorNode.click();
  downloadAnchorNode.remove();
}

export { parseMapFromSocket, parseMapFromLcm, normalizeList, downloadMapFile };
