# Convert to a python class
from typing import List, Union

# Convert to a python class


# accelerometer
class Accelerometer:

    class Acceleration:
        v: List[float]
        status: int

    acceleration: Acceleration
    version: int
    sensor: int
    type: int
    timestamp: int
    source: str
    uncalibratedDEPRECATED: bool


# androidLog
class AndroidLog:

    class MessageDetail:
        MESSAGE: str
        PRIORITY: str
        SYSLOG_FACILITY: str
        _BOOT_ID: str
        _HOSTNAME: str
        _MACHINE_ID: str
        _SOURCE_MONOTONIC_TIMESTAMP: str
        _TRANSPORT: str

    id: int
    ts: int
    priority: int
    pid: int
    tid: int
    message: MessageDetail


# cameraOdometry
class CameraOdometry:
    trans: List[float]
    rot: List[float]
    transStd: List[float]
    rotStd: List[float]
    frameId: int
    timestampEof: int
    wideFromDeviceEuler: List[float]
    wideFromDeviceEulerStd: List[float]
    roadTransformTrans: List[float]
    roadTransformTransStd: List[float]


class _CANMessage:
    """Use the CAN class instead"""

    address: int
    busTime: int
    dat: bytes  # Note: bytes type is used for binary data like b'\x00\x1e...'
    src: int


# can
class CAN(List[_CANMessage]):
    pass


# carControl
class CarControl:
    class CruiseControl:
        cancel: bool
        resume: bool
        speedOverrideDEPRECATED: float
        accelOverrideDEPRECATED: float
        override: bool

    class HudControl:
        speedVisible: bool
        setSpeed: float
        lanesVisible: bool
        leadVisible: bool
        visualAlert: str
        audibleAlert: str
        rightLaneVisible: bool
        leftLaneVisible: bool
        rightLaneDepart: bool
        leftLaneDepart: bool

    class Actuators:
        gas: float
        brake: float
        steer: float
        steeringAngleDeg: float
        accel: float
        longControlState: str
        speed: float
        curvature: float
        steerOutputCan: float

    enabled: bool
    gasDEPRECATED: float
    brakeDEPRECATED: float
    steeringTorqueDEPRECATED: float
    cruiseControl: CruiseControl
    hudControl: HudControl
    actuators: Actuators
    activeDEPRECATED: bool
    rollDEPRECATED: float
    pitchDEPRECATED: float
    latActive: bool
    longActive: bool
    orientationNED: List[float]
    angularVelocity: List[float]
    leftBlinker: bool
    rightBlinker: bool


# carParams
class CarParams:

    class LongitudinalTuning:
        kpBP: List[float]
        kpV: List[float]
        kiBP: List[float]
        kiV: List[float]
        deadzoneBP: List[float]
        deadzoneV: List[float]
        kf: float

    class LateralTuning:
        class PID:
            kf: float

        pid: PID

    class CarFw:
        ecu: str
        fwVersion: bytes
        address: int
        subAddress: int
        responseAddress: int
        request: List[bytes]
        brand: str
        bus: int
        logging: bool
        obdMultiplexing: bool

    class SafetyConfig:
        safetyModel: str
        safetyParamDEPRECATED: int
        safetyParam2DEPRECATED: int
        safetyParam: int

    carName: str
    carFingerprint: str
    enableGasInterceptor: bool
    pcmCruise: bool
    enableCameraDEPRECATED: bool
    enableDsu: bool
    enableApgsDEPRECATED: bool
    minEnableSpeed: float
    minSteerSpeed: float
    safetyModelDEPRECATED: str
    safetyParamDEPRECATED: int
    mass: float
    wheelbase: float
    centerToFront: float
    steerRatio: float
    steerRatioRear: float
    rotationalInertia: float
    tireStiffnessFront: float
    tireStiffnessRear: float
    longitudinalTuning: LongitudinalTuning
    lateralTuning: LateralTuning
    steerLimitAlert: bool
    vEgoStopping: float
    directAccelControlDEPRECATED: bool
    stoppingControl: bool
    startAccel: float
    steerRateCostDEPRECATED: float
    steerControlType: str
    radarUnavailable: bool
    steerActuatorDelay: float
    openpilotLongitudinalControl: bool
    carVin: str
    isPandaBlackDEPRECATED: bool
    dashcamOnly: bool
    safetyModelPassiveDEPRECATED: str
    transmissionType: str
    carFw: List[CarFw]
    radarTimeStep: float
    communityFeatureDEPRECATED: bool
    steerLimitTimer: float
    fingerprintSource: str
    networkLocation: str
    minSpeedCanDEPRECATED: float
    stoppingDecelRate: float
    startingAccelRateDEPRECATED: float
    maxSteeringAngleDegDEPRECATED: float
    fuzzyFingerprint: bool
    enableBsm: bool
    hasStockCameraDEPRECATED: bool
    longitudinalActuatorDelayUpperBound: float
    vEgoStarting: float
    stopAccel: float
    longitudinalActuatorDelayLowerBound: float
    safetyConfigs: List[SafetyConfig]
    wheelSpeedFactor: float
    flags: int
    alternativeExperience: int
    notCar: bool
    maxLateralAccel: float
    autoResumeSng: bool
    startingState: bool
    experimentalLongitudinalAvailable: bool
    tireStiffnessFactor: float
    passive: bool


# carState
class CarState:

    class CruiseState:
        enabled: bool
        speed: float
        available: bool
        speedOffset: float
        standstill: bool
        nonAdaptive: bool
        speedCluster: float

    class Event:
        # examples of event names:
        # doorOpen, seatbeltNotLatched, wrongGear, wrongCarMode, parkBrake, pcmDisable
        name: str
        enable: bool
        noEntry: bool
        warning: bool
        userDisable: bool
        softDisable: bool
        immediateDisable: bool
        preEnable: bool
        permanent: bool
        overrideLongitudinal: bool
        overrideLateral: bool

    vEgo: float
    gas: float
    gasPressed: bool
    brake: float
    brakePressed: bool
    steeringAngleDeg: float
    steeringTorque: float
    steeringPressed: bool
    cruiseState: CruiseState
    events: list[Event]
    gearShifter: str
    steeringRateDeg: float
    aEgo: float
    vEgoRaw: float
    standstill: bool
    brakeLightsDEPRECATED: bool
    leftBlinker: bool
    rightBlinker: bool
    yawRate: float
    genericToggle: bool
    doorOpen: bool
    seatbeltUnlatched: bool
    canValid: bool
    steeringTorqueEps: float
    clutchPressed: bool
    steeringRateLimitedDEPRECATED: bool
    stockAeb: bool
    stockFcw: bool
    espDisabled: bool
    leftBlindspot: bool
    rightBlindspot: bool
    steerFaultTemporary: bool
    steerFaultPermanent: bool
    steeringAngleOffsetDeg: float
    brakeHoldActive: bool
    parkingBrake: bool
    canTimeout: bool
    fuelGauge: float
    accFaulted: bool
    charging: bool
    vEgoCluster: float
    regenBraking: bool
    engineRpm: float
    carFaultedNonCritical: bool


# controlsState
class ControlsState:

    class LateralControlState:
        class PidState:
            active: bool
            steeringAngleDeg: float
            steeringRateDeg: float
            angleError: float
            p: float
            i: float
            f: float
            output: float
            saturated: bool
            steeringAngleDesiredDeg: float

        pidState: PidState

    vEgoDEPRECATED: float
    aEgoDEPRECATED: float
    vPid: float
    vTargetLead: float
    upAccelCmd: float
    uiAccelCmd: float
    yActualDEPRECATED: float
    yDesDEPRECATED: float
    upSteerDEPRECATED: float
    uiSteerDEPRECATED: float
    aTargetMinDEPRECATED: float
    aTargetMaxDEPRECATED: float
    jerkFactorDEPRECATED: float
    angleSteersDEPRECATED: float
    hudLeadDEPRECATED: int
    cumLagMs: float
    canMonoTimeDEPRECATED: int
    radarStateMonoTimeDEPRECATED: int
    mdMonoTimeDEPRECATED: int
    enabled: bool
    steerOverrideDEPRECATED: bool
    vCruise: float
    rearViewCamDEPRECATED: bool
    alertText1: str
    alertText2: str
    awarenessStatusDEPRECATED: float
    angleModelBiasDEPRECATED: float
    longitudinalPlanMonoTime: int
    steeringAngleDesiredDegDEPRECATED: float
    longControlState: str
    state: str
    vEgoRawDEPRECATED: float
    ufAccelCmd: float
    ufSteerDEPRECATED: float
    aTarget: float
    active: bool
    curvature: float
    alertStatus: str
    alertSize: str
    gpsPlannerActiveDEPRECATED: bool
    engageable: bool
    alertBlinkingRate: float
    driverMonitoringOnDEPRECATED: bool
    alertType: str
    vCurvatureDEPRECATED: float
    decelForTurnDEPRECATED: bool
    startMonoTime: int
    mapValidDEPRECATED: bool
    lateralPlanMonoTime: int
    forceDecel: bool
    lateralControlState: LateralControlState
    decelForModelDEPRECATED: bool
    alertSound: str
    canErrorCounter: int
    desiredCurvature: float
    desiredCurvatureRateDEPRECATED: float
    vCruiseCluster: float
    experimentalMode: bool


# deviceState
class DeviceState:

    class NetworkInfo:
        technology: str
        operator: str
        band: str
        channel: int
        extra: str
        state: str

    class NetworkStats:
        wwanTx: int
        wwanRx: int

    cpu0DEPRECATED: int
    cpu1DEPRECATED: int
    cpu2DEPRECATED: int
    cpu3DEPRECATED: int
    memDEPRECATED: int
    gpuDEPRECATED: int
    batDEPRECATED: int
    freeSpacePercent: float
    batteryPercentDEPRECATED: int
    fanSpeedPercentDesired: int
    started: bool
    usbOnlineDEPRECATED: bool
    startedMonoTime: int
    thermalStatus: str
    batteryCurrentDEPRECATED: int
    batteryVoltageDEPRECATED: int
    chargingErrorDEPRECATED: bool
    chargingDisabledDEPRECATED: bool
    memoryUsagePercent: int
    cpuUsagePercentDEPRECATED: int
    pa0DEPRECATED: int
    networkType: str
    offroadPowerUsageUwh: int
    networkStrength: str
    carBatteryCapacityUwh: int
    cpuTempC: list[float]
    gpuTempC: list[float]
    memoryTempC: float
    batteryTempCDEPRECATED: float
    ambientTempC: float
    networkInfo: NetworkInfo
    lastAthenaPingTime: int
    gpuUsagePercent: int
    cpuUsagePercent: list[int]
    nvmeTempC: list
    modemTempC: list[float]
    screenBrightnessPercent: int
    pmicTempC: list[float]
    powerDrawW: float
    networkMetered: bool
    somPowerDrawW: float
    networkStats: NetworkStats
    maxTempC: float


# driverCameraState
class DriverCameraState:
    frameId: int
    encodeId: int
    timestampEof: int
    frameLengthDEPRECATED: int
    integLines: int
    globalGainDEPRECATED: int
    frameType: str
    timestampSof: int
    lensPosDEPRECATED: int
    lensSagDEPRECATED: float
    lensErrDEPRECATED: float
    lensTruePosDEPRECATED: float
    gain: float
    recoverStateDEPRECATED: int
    highConversionGain: bool
    measuredGreyFraction: float
    targetGreyFraction: float
    processingTime: float
    frameIdSensor: int
    sensor: str
    exposureValPercent: float
    requestId: int


# driverEncodeIdx
class DriverEncodeIdx:
    frameId: int
    type: str
    encodeId: int
    segmentNum: int
    segmentId: int
    segmentIdEncode: int
    timestampSof: int
    timestampEof: int
    flags: int
    len: int


# driverMonitoringState
class DriverMonitoringState:
    events: list
    faceDetected: bool
    isDistracted: bool
    awarenessStatus: float
    isRHD: bool
    rhdCheckedDEPRECATED: bool
    posePitchOffset: float
    posePitchValidCount: int
    poseYawOffset: float
    poseYawValidCount: int
    stepChange: float
    awarenessActive: float
    awarenessPassive: float
    isLowStd: bool
    hiStdCount: int
    isPreviewDEPRECATED: bool
    isActiveMode: bool
    distractedType: int


# driverStateV2
class DriverStateV2:

    class DriverInfo:
        faceOrientation: list[float]
        faceOrientationStd: list[float]
        facePosition: list[float]
        facePositionStd: list[float]
        faceProb: float
        leftEyeProb: float
        rightEyeProb: float
        leftBlinkProb: float
        rightBlinkProb: float
        sunglassesProb: float
        occludedProb: float
        readyProb: list[float]
        notReadyProb: list[float]

    frameId: int
    modelExecutionTime: float
    dspExecutionTime: float
    rawPredictions: (
        bytes  # Note: This holds raw binary data and may need special handling
    )
    poorVisionProb: float
    wheelOnRightProb: float
    leftDriverData: DriverInfo
    rightDriverData: DriverInfo


# gpsLocation
class GpsLocation:
    flags: int
    latitude: float
    longitude: float
    altitude: float
    speed: float
    bearingDeg: float
    accuracy: float
    unixTimestampMillis: int
    source: str
    vNED: list[float]
    verticalAccuracy: float
    bearingAccuracyDeg: float
    speedAccuracy: float


# gyroscope
class Gyroscope:

    class GyroUncalibrated:
        v: list[float]
        status: int

    gyroUncalibrated: GyroUncalibrated
    version: int
    sensor: int
    type: int
    timestamp: int
    source: str
    uncalibratedDEPRECATED: bool


# initData
class InitData:

    class KernelArgs:
        kernelArgs: List[str]

    class Params:
        class Entry:
            key: str
            value: bytes

        entries: List[Entry]

    class Commands:
        class Entry:
            key: str
            value: bytes

        entries: List[Entry]

    kernelArgs: KernelArgs
    dongleId: str
    deviceType: str
    version: str
    dirty: bool
    gitCommit: str
    gitBranch: str
    passive: bool
    gitRemote: str
    kernelVersion: str
    params: Params
    osVersion: str
    commands: Commands
    wallTimeNanos: int


# lateralPlanDEPRECATED
class LateralPlanDEPRECATED:

    laneWidthDEPRECATED: float
    cProbDEPRECATED: float
    lProbDEPRECATED: float
    rProbDEPRECATED: float
    steeringAngleDegDEPRECATED: float
    mpcSolutionValid: bool
    paramsValidDEPRECATED: bool
    angleOffsetDegDEPRECATED: float
    modelValidDEPRECATED: bool
    steeringRateDegDEPRECATED: float
    sensorValidDEPRECATED: bool
    commIssueDEPRECATED: bool
    posenetValidDEPRECATED: bool
    desire: str
    laneChangeState: str
    laneChangeDirection: str
    dPathPoints: List[float]
    dProbDEPRECATED: float
    curvatureDEPRECATED: float
    curvatureRateDEPRECATED: float
    rawCurvatureDEPRECATED: float
    rawCurvatureRateDEPRECATED: float
    psis: List[float]
    curvatures: List[float]
    curvatureRates: List[float]
    useLaneLines: bool
    solverExecutionTime: float
    modelMonoTime: int
    solverCost: float


# liveCalibration
class LiveCalibration:

    calStatusDEPRECATED: int
    calCycle: int
    calPerc: int
    rpyCalib: List[float]
    rpyCalibSpread: List[float]
    validBlocks: int
    wideFromDeviceEuler: List[float]
    calStatus: str
    height: List[float]


# liveLocationKalman
class LiveLocationKalman:

    class ValueGroup:
        value: List[float]
        std: List[float]
        valid: bool

    positionECEF: ValueGroup
    positionGeodetic: ValueGroup
    velocityECEF: ValueGroup
    velocityNED: ValueGroup
    velocityDevice: ValueGroup
    accelerationDevice: ValueGroup
    orientationECEF: ValueGroup
    orientationNED: ValueGroup
    angularVelocityDevice: ValueGroup
    calibratedOrientationNED: ValueGroup
    velocityCalibrated: ValueGroup
    accelerationCalibrated: ValueGroup
    angularVelocityCalibrated: ValueGroup
    calibratedOrientationECEF: ValueGroup
    gpsWeek: int
    gpsTimeOfWeek: float
    status: str
    unixTimestampMillis: int
    inputsOK: bool
    posenetOK: bool
    gpsOK: bool
    sensorsOK: bool
    deviceStable: bool
    timeSinceReset: float
    excessiveResets: bool
    timeToFirstFix: float


# liveParameters
class VehicleState:

    valid: bool
    gyroBias: float
    angleOffsetDeg: float
    angleOffsetAverageDeg: float
    stiffnessFactor: float
    steerRatio: float
    sensorValid: bool
    yawRateDEPRECATED: float
    posenetSpeed: float
    posenetValid: bool
    angleOffsetFastStd: float
    angleOffsetAverageStd: float
    stiffnessFactorStd: float
    steerRatioStd: float
    roll: float


# liveTorqueParameters
class LiveTorqueParameters:

    liveValid: bool
    latAccelFactorRaw: float
    latAccelOffsetRaw: float
    frictionCoefficientRaw: float
    latAccelFactorFiltered: float
    latAccelOffsetFiltered: float
    frictionCoefficientFiltered: float
    totalBucketPoints: float
    decay: float
    maxResets: float
    version: int
    useParams: bool


# liveTracks
class LiveTracks(List):
    # Do not know the fields here yet as all the lists
    # in the example data are empty
    pass


# logMessage
class LogMessage(str):
    # Is a json string
    pass


# longitudinalPlan
class LongitudinalPlan:

    lateralValidDEPRECATED: bool
    longitudinalValidDEPRECATED: bool
    vTargetDEPRECATED: float
    aTargetMinDEPRECATED: float
    aTargetMaxDEPRECATED: float
    jerkFactorDEPRECATED: float
    hasLead: bool
    fcw: bool
    modelMonoTime: int
    radarStateMonoTimeDEPRECATED: int
    laneWidthDEPRECATED: float
    vTargetFutureDEPRECATED: float
    longitudinalPlanSource: str
    vCruiseDEPRECATED: float
    aCruiseDEPRECATED: float
    aTargetDEPRECATED: float
    gpsPlannerActiveDEPRECATED: bool
    vMaxDEPRECATED: float
    vCurvatureDEPRECATED: float
    decelForTurnDEPRECATED: bool
    hasLeftLaneDEPRECATED: bool
    hasRightLaneDEPRECATED: bool
    mapValidDEPRECATED: bool
    vStartDEPRECATED: float
    aStartDEPRECATED: float
    radarValidDEPRECATED: bool
    processingDelay: float
    radarCanErrorDEPRECATED: bool
    commIssueDEPRECATED: bool
    accels: List[float]
    speeds: List[float]
    jerks: List[float]
    solverExecutionTime: float
    personality: str


# magnetometer
class Magnetometer:

    class MagneticUncalibrated:
        v: List[float]
        status: int

    magneticUncalibrated: MagneticUncalibrated
    version: int
    sensor: int
    type: int
    timestamp: int
    source: str
    uncalibratedDEPRECATED: bool


# managerState
class ManagerState:

    class Process:
        name: str
        pid: int
        running: bool
        exitCode: int
        shouldBeRunning: bool

    processes: List[Process]


# mapRenderState
class MapRenderState:

    locationMonoTime: int
    renderTime: float
    frameId: int


# microphone
class Microphone:

    soundPressure: float
    soundPressureWeightedDb: float
    filteredSoundPressureWeightedDb: float
    soundPressureWeighted: float


# modelV2
class ModelV2:

    class Position:
        x: List[float]
        y: List[float]
        z: List[float]
        t: List[float]
        xStd: List[float]
        yStd: List[float]
        zStd: List[float]

    class TimestampedValues:
        x: List[float]
        y: List[float]
        z: List[float]
        t: List[float]

    class LeadsV3:
        prob: float
        probTime: float
        t: List[float]
        x: List[float]
        xStd: List[float]
        y: List[float]
        yStd: List[float]
        v: List[float]
        vStd: List[float]
        a: List[float]
        aStd: List[float]

    class TemporalPose:
        trans: List[float]
        rot: List[float]
        transStd: List[float]
        rotStd: List[float]

    class DisengagePredictions:
        t: List[float]
        brakeDisengageProbs: List[float]
        gasDisengageProbs: List[float]
        steerOverrideProbs: List[float]
        brake3MetersPerSecondSquaredProbs: List[float]
        brake4MetersPerSecondSquaredProbs: List[float]
        brake5MetersPerSecondSquaredProbs: List[float]

    class Meta:
        engagedProb: float
        desirePrediction: List[float]
        brakeDisengageProbDEPRECATED: float
        gasDisengageProbDEPRECATED: float
        steerOverrideProbDEPRECATED: float
        desireState: List[float]
        disengagePredictions: DisengagePredictions
        hardBrakePredicted: bool
        laneChangeState: str
        laneChangeDirection: str

    class LateralPlannerSolutionDEPRECATED:
        x: List[float]
        y: List[float]
        yaw: List[float]
        yawRate: List[float]
        xStd: List[float]
        yStd: List[float]
        yawStd: List[float]
        yawRateStd: List[float]

    frameId: int
    frameAge: int
    frameDropPerc: float
    timestampEof: int
    position: Position
    orientation: TimestampedValues
    velocity: TimestampedValues
    orientationRate: TimestampedValues
    laneLines: List[TimestampedValues]
    laneLineProbs: List[float]
    roadEdges: List[TimestampedValues]
    roadEdgeStds: List[float]
    modelExecutionTime: float
    gpuExecutionTime: float
    leadsV3: List[LeadsV3]
    acceleration: TimestampedValues
    frameIdExtra: int
    temporalPose: TemporalPose
    navEnabled: bool
    confidence: str
    locationMonoTime: int
    lateralPlannerSolutionDEPRECATED: LateralPlannerSolutionDEPRECATED
    laneLineStds: List[float]
    meta: Meta


# navInstruction
class NavInstruction:

    maneuverDistance: float
    distanceRemaining: float
    timeRemaining: float
    timeRemainingTypical: float
    showFull: bool
    speedLimit: float
    speedLimitSign: str


# navModel
class NavModel:

    class Position:
        x: List[float]
        y: List[float]
        xStd: List[float]
        yStd: List[float]

    frameId: int
    modelExecutionTime: float
    dspExecutionTime: float
    features: List[float]
    position: Position
    desirePrediction: List[float]
    locationMonoTime: int


# navThumbnail
class ImageData:
    frameId: int
    timestampEof: int
    thumbnail: bytes  # Binary data for the thumbnail


class _OnroadEventItem:

    name: str
    enable: bool
    noEntry: bool
    warning: bool
    userDisable: bool
    softDisable: bool
    immediateDisable: bool
    preEnable: bool
    permanent: bool
    overrideLongitudinal: bool
    overrideLateral: bool


# onroadEvents
class OnroadEvenets(List[_OnroadEventItem]):
    pass


class _PandaStateItem:

    class CanState:
        busOff: bool
        busOffCnt: int
        errorWarning: bool
        errorPassive: bool
        lastError: str
        lastStoredError: str
        lastDataError: str
        lastDataStoredError: str
        receiveErrorCnt: int
        transmitErrorCnt: int
        totalErrorCnt: int
        totalTxLostCnt: int
        totalRxLostCnt: int
        totalTxCnt: int
        totalRxCnt: int
        totalFwdCnt: int
        canSpeed: int
        canDataSpeed: int
        canfdEnabled: bool
        brsEnabled: bool
        canfdNonIso: bool
        irq0CallRate: int
        irq1CallRate: int
        irq2CallRate: int
        canCoreResetCnt: int

    voltage: int
    current: int
    ignitionLine: bool
    controlsAllowed: bool
    gasInterceptorDetectedDEPRECATED: bool
    startedSignalDetectedDEPRECATED: bool
    hasGpsDEPRECATED: bool
    rxBufferOverflow: int
    txBufferOverflow: int
    gmlanSendErrs: int
    pandaType: str
    fanSpeedRpmDEPRECATED: int
    usbPowerModeDEPRECATED: str
    ignitionCan: bool
    safetyModel: str
    faultStatus: str
    powerSaveEnabled: bool
    uptime: int
    faults: list
    safetyRxInvalid: int
    safetyParamDEPRECATED: int
    harnessStatus: str
    heartbeatLost: bool
    alternativeExperience: int
    safetyTxBlocked: int
    interruptLoad: float
    safetyParam2DEPRECATED: int
    safetyParam: int
    fanPower: int
    canState0: CanState
    canState1: CanState
    canState2: CanState
    safetyRxChecksInvalid: bool
    spiChecksumErrorCount: int
    fanStallCount: int
    sbu1Voltage: float
    sbu2Voltage: float


# pandaStates
class PandaStates(List[_PandaStateItem]):
    pass


# peripheralState
class PeripheralState:

    pandaType: str
    voltage: int
    current: int
    fanSpeedRpm: int
    usbPowerModeDEPRECATED: str


# procLog
class ProcLog:

    class CpuTime:
        cpuNum: int
        user: float
        nice: float
        system: float
        idle: float
        iowait: float
        irq: float
        softirq: float

    class Mem:
        total: int
        free: int
        available: int
        buffers: int
        cached: int
        active: int
        inactive: int
        shared: int

    class Proc:
        pid: int
        name: str
        state: int
        ppid: int
        cpuUser: float
        cpuSystem: float
        cpuChildrenUser: float
        cpuChildrenSystem: float
        priority: int
        nice: int
        numThreads: int
        startTime: float
        memVms: int
        memRss: int
        processor: int
        cmdline: List[str]
        exe: str

    cpuTimes: List[CpuTime]
    mem: Mem
    procs: List[Proc]


# qRoadEncodeIdx
class QRoadEncodeIdx:

    frameId: int
    type: str
    encodeId: int
    segmentNum: int
    segmentId: int
    segmentIdEncode: int
    timestampSof: int
    timestampEof: int
    flags: int
    len: int


# qcomGnss
class QcomGnss:

    class MeasurementReport:
        source: str
        fCount: int
        gpsWeek: int
        glonassCycleNumber: int
        glonassNumberOfDays: int
        milliseconds: int
        timeBias: float
        clockTimeUncertainty: float
        clockFrequencyBias: float
        clockFrequencyUncertainty: float

        class Sv:
            svId: int
            glonassFrequencyIndex: int
            observationState: str
            observations: int
            goodObservations: int
            gpsParityErrorCount: int
            glonassHemmingErrorCount: int
            filterStages: int
            carrierNoise: int
            latency: int
            predetectInterval: int
            postdetections: int
            unfilteredMeasurementIntegral: int
            unfilteredMeasurementFraction: float
            unfilteredTimeUncertainty: float
            unfilteredSpeed: float
            unfilteredSpeedUncertainty: float

            class MeasurementStatus:
                subMillisecondIsValid: bool
                subBitTimeIsKnown: bool
                satelliteTimeIsKnown: bool
                bitEdgeConfirmedFromSignal: bool
                measuredVelocity: bool
                fineOrCoarseVelocity: bool
                lockPointValid: bool
                lockPointPositive: bool
                lastUpdateFromDifference: bool
                lastUpdateFromVelocityDifference: bool
                strongIndicationOfCrossCorelation: bool
                tentativeMeasurement: bool
                measurementNotUsable: bool
                sirCheckIsNeeded: bool
                probationMode: bool
                glonassMeanderBitEdgeValid: bool
                glonassTimeMarkValid: bool
                gpsRoundRobinRxDiversity: bool
                gpsRxDiversity: bool
                gpsLowBandwidthRxDiversityCombined: bool
                gpsHighBandwidthNu4: bool
                gpsHighBandwidthNu8: bool
                gpsHighBandwidthUniform: bool
                multipathIndicator: bool
                imdJammingIndicator: bool
                lteB13TxJammingIndicator: bool
                freshMeasurementIndicator: bool
                multipathEstimateIsValid: bool
                directionIsValid: bool

            measurementStatus: MeasurementStatus
            multipathEstimate: float
            azimuth: float
            elevation: float
            carrierPhaseCyclesIntegral: int
            carrierPhaseCyclesFraction: int
            fineSpeed: float
            fineSpeedUncertainty: float
            cycleSlipCount: int

        sv: List[Sv]

    measurementReport: MeasurementReport
    logTs: int


# radarState
class RadarState:

    class LeadData:
        dRel: float
        yRel: float
        vRel: float
        aRel: float
        vLead: float
        aLeadDEPRECATED: float
        dPath: float
        vLat: float
        vLeadK: float
        aLeadK: float
        fcw: bool
        status: bool
        aLeadTau: float
        modelProb: float
        radar: bool
        radarTrackId: int

    angleOffsetDEPRECATED: float
    calStatusDEPRECATED: int
    leadOne: LeadData
    leadTwo: LeadData
    cumLagMs: float
    mdMonoTime: int
    ftMonoTimeDEPRECATED: int
    calCycleDEPRECATED: int
    calPercDEPRECATED: int
    carStateMonoTime: int
    radarErrors: list


# roadCameraState
class RoadCameraState:

    frameId: int
    encodeId: int
    timestampEof: int
    frameLengthDEPRECATED: int
    integLines: int
    globalGainDEPRECATED: int
    frameType: str
    timestampSof: int
    lensPosDEPRECATED: int
    lensSagDEPRECATED: float
    lensErrDEPRECATED: float
    lensTruePosDEPRECATED: float
    gain: float
    recoverStateDEPRECATED: int
    highConversionGain: bool
    measuredGreyFraction: float
    targetGreyFraction: float
    processingTime: float
    frameIdSensor: int
    sensor: str
    exposureValPercent: float
    requestId: int


# roadEncodeIdx
class RoadEncodeIdx:

    frameId: int
    type: str
    encodeId: int
    segmentNum: int
    segmentId: int
    segmentIdEncode: int
    timestampSof: int
    timestampEof: int
    flags: int
    len: int


# sentinel
class Sentinel:

    type: str
    signal: int


# temperatureSensor
class TemperatureSensor:

    temperature: float
    version: int
    sensor: int
    type: int
    timestamp: int
    source: str
    uncalibratedDEPRECATED: bool


# thumbnail
class Thumbnail:

    frameId: int
    timestampEof: int
    thumbnail: bytes


# uiDebug
class UiDebug:
    drawTimeMillis: float


# uiPlan
class UiPlan:

    class Position:
        x: List[float]
        y: List[float]
        z: List[float]

    position: Position
    accel: List[float]
    frameId: int


# wideRoadCameraState
class WideRoadCameraState:

    frameId: int
    encodeId: int
    timestampEof: int
    frameLengthDEPRECATED: int
    integLines: int
    globalGainDEPRECATED: int
    frameType: str
    timestampSof: int
    lensPosDEPRECATED: int
    lensSagDEPRECATED: float
    lensErrDEPRECATED: float
    lensTruePosDEPRECATED: float
    gain: float
    recoverStateDEPRECATED: int
    highConversionGain: bool
    measuredGreyFraction: float
    targetGreyFraction: float
    processingTime: float
    frameIdSensor: int
    sensor: str
    exposureValPercent: float
    requestId: int


# wideRoadEncodeIdx
class WideRoadEncodeIdx:

    frameId: int
    type: str
    encodeId: int
    segmentNum: int
    segmentId: int
    segmentIdEncode: int
    timestampSof: int
    timestampEof: int
    flags: int
    len: int


class Clocks:
    bootTimeNanosDEPRECATED: int
    monotonicNanosDEPRECATED: int
    monotonicRawNanosDEPRECATD: int
    wallTimeNanos: int
    modemUptimeMillisDEPRECATED: int


class _SendcanItem:

    class SendCan:
        address: int
        busTime: int
        dat: Union[
            bytes, str
        ]  # Adjust based on the actual use case (bytes or a representation of it)
        src: int

    sendcan: List[SendCan]
    logMonoTime: int
    valid: bool


class Sendcan(List[_SendcanItem]):
    pass


class ErrorLogMessage(str):
    """Is a json string containing msg, ctx ++."""

    pass
