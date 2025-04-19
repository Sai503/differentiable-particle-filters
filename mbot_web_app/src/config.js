
var config = {
    // Connection info.
    HOST: "localhost",
    PORT: 5000,
    ENDPOINT: "",
    CONNECT_PERIOD: 5000,     // ms

    //Default Values
    ROBOT_START_X: 400,
    ROBOT_START_Y: 400,
    CELL_START_SIZE: 0.025,

    // Sim info
    MAP_DISPLAY_WIDTH: 500,      // px
    MAP_DISPLAY_HEIGHT: 500,     // px
    CANVAS_DISPLAY_WIDTH: 800,   // px
    CANVAS_DISPLAY_HEIGHT: 800,  // px
    ROBOT_SIZE: 0.205,           // m, diameter
    ROBOT_DEFAULT_SIZE: 100,     // px
    MAP_UPDATE_PERIOD: 250,      // ms
    STALE_MAP_COUNT: 40,         // If we haven't gotten a map this many times, map is stale.

    // Driving info
    ANG_VEL_MULTIPLIER: 5.0,  // Scale to multiply values [0-1] for angular velocity.
    DRIVE_CMD_RATE: 100,      // Rate for sending drive commands when active.

    // Display info
    MAP_COLOUR_HIGH: "#000000",      // Black
    MAP_COLOUR_LOW: "#ffffff",       // White
    FIELD_COLOUR_HIGH: "#444444",    // White
    FIELD_COLOUR_LOW: "#ffffff",     // Grey
    FIELD_ALPHA: "99",
    PATH_COLOUR: "#00B2A9",            // Taubman Teal
    VISITED_CELL_COLOUR: "#989C97",    // Angell Hall Ash
    CLICKED_CELL_COLOUR: "#FFCB05",    // Maize
    GOAL_CELL_COLOUR: "#00ff00",
    BAD_GOAL_COLOUR: "#ff0000",
    SMALL_CELL_SCALE: 0.8,
    CELL_SIZE: 4,

    // Modes for SLAM.
    slam_mode: {
        INVALID: -1,
        MAPPING_ONLY: 0,
        ACTION_ONLY: 1,
        LOCALIZATION_ONLY: 2,
        FULL_SLAM: 3,
        IDLE: 99,
    },

    // MBot Channels.
    POSE_CHANNEL: "SLAM_POSE",
    LIDAR_CHANNEL: "LIDAR",
    PARTICLE_CHANNEL: "SLAM_PARTICLES",
    PATH_CHANNEL: "CONTROLLER_PATH",
    SLAM_MODE_CHANNEL: "SLAM_STATUS",
    SLAM_MAP_CHANNEL: "SLAM_MAP",
};

export default config;
