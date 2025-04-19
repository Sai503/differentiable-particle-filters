
function normalizeAngle(angle) {
  // Normalizes an angle from -PI to PI.
  let result = (angle + Math.PI) % (2.0 * Math.PI);
  if (result <= 0.0) return result + Math.PI;
  return result - Math.PI;
}

function isDeepEqual(data1, data2) {
  if (data1 === data2) {
    return true;
  }

  if (!data1 || !data2) {
    return false;
  }

  if (data1.length !== data2.length) {
    return false;
  }

  const chs1 = data1.map((item, index) => (item.channel));
  const chs2 = data2.map((item, index) => (item.channel));

  for (const key of chs1) {
    if (!chs2.includes(key)) {
      return false;
    }
  }

  for (const key of chs2) {
    if (!chs1.includes(key)) {
      return false;
    }
  }

  return true;
}

export { normalizeAngle, isDeepEqual };
