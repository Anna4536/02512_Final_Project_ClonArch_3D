const fs = require("fs");
const { isoLines } = require("../../dist/marchingsquares-isolines.js");

const { matrix, threshold, options } = JSON.parse(fs.readFileSync(0, "utf8"));
const contours = isoLines(matrix, threshold, options || {});
process.stdout.write(JSON.stringify(contours));