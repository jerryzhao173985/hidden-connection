/**
 * Hidden Connections - Semantic Nebula Visualization
 * A cinematic, animated visualization of semantic relationships
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

const CONFIG = {
    // Cluster colors (nebula hues) - richer, more saturated
    colors: [
        { r: 64, g: 224, b: 208, name: 'Turquoise' },   // Turquoise
        { r: 255, g: 99, b: 132, name: 'Rose' },        // Rose
        { r: 180, g: 150, b: 255, name: 'Lavender' },   // Lavender
        { r: 255, g: 206, b: 86, name: 'Amber' },       // Amber
        { r: 102, g: 255, b: 178, name: 'Mint' },       // Mint
        { r: 255, g: 159, b: 64, name: 'Tangerine' },   // Tangerine
    ],

    // Star settings
    star: {
        baseRadius: 2.5,
        hoverRadius: 5,
        neighborRadius: 3.5,
        glowRadius: 20,
        breathSpeed: 0.0015,
        breathAmount: 0.2,
        twinkleSpeed: 0.008,
        twinkleAmount: 0.15,
    },

    // Link settings
    link: {
        baseAlpha: 0.06,
        hoverAlpha: 0.5,
        pulseSpeed: 0.002,
        travelSpeed: 0.003,
        baseWidth: 0.6,
        hoverWidth: 2,
        particleCount: 2,
    },

    // Nebula settings
    nebula: {
        layers: 4,
        noiseScale: 0.0015,
        noiseSpeed: 0.0002,
        baseAlpha: 0.035,
        drift: 0.3,
    },

    // Background stars
    bgStars: {
        count: 150,
        minSize: 0.5,
        maxSize: 1.5,
        twinkleSpeed: 0.005,
    },

    // Hover detection
    hoverThreshold: 30,

    // Animation
    transitionSpeed: 0.08,
};

// ============================================================================
// SIMPLEX NOISE (for nebula animation)
// ============================================================================

class SimplexNoise {
    constructor(seed = Math.random()) {
        this.p = new Uint8Array(256);
        for (let i = 0; i < 256; i++) this.p[i] = i;

        let n = seed * 256;
        for (let i = 255; i > 0; i--) {
            n = (n * 16807) % 2147483647;
            const j = n % (i + 1);
            [this.p[i], this.p[j]] = [this.p[j], this.p[i]];
        }

        this.perm = new Uint8Array(512);
        for (let i = 0; i < 512; i++) this.perm[i] = this.p[i & 255];
    }

    noise2D(x, y) {
        const F2 = 0.5 * (Math.sqrt(3) - 1);
        const G2 = (3 - Math.sqrt(3)) / 6;

        const s = (x + y) * F2;
        const i = Math.floor(x + s);
        const j = Math.floor(y + s);

        const t = (i + j) * G2;
        const X0 = i - t;
        const Y0 = j - t;
        const x0 = x - X0;
        const y0 = y - Y0;

        const i1 = x0 > y0 ? 1 : 0;
        const j1 = x0 > y0 ? 0 : 1;

        const x1 = x0 - i1 + G2;
        const y1 = y0 - j1 + G2;
        const x2 = x0 - 1 + 2 * G2;
        const y2 = y0 - 1 + 2 * G2;

        const ii = i & 255;
        const jj = j & 255;

        const grad = (hash, x, y) => {
            const h = hash & 7;
            const u = h < 4 ? x : y;
            const v = h < 4 ? y : x;
            return ((h & 1) ? -u : u) + ((h & 2) ? -2 * v : 2 * v);
        };

        let n0 = 0, n1 = 0, n2 = 0;

        let t0 = 0.5 - x0 * x0 - y0 * y0;
        if (t0 >= 0) {
            t0 *= t0;
            n0 = t0 * t0 * grad(this.perm[ii + this.perm[jj]], x0, y0);
        }

        let t1 = 0.5 - x1 * x1 - y1 * y1;
        if (t1 >= 0) {
            t1 *= t1;
            n1 = t1 * t1 * grad(this.perm[ii + i1 + this.perm[jj + j1]], x1, y1);
        }

        let t2 = 0.5 - x2 * x2 - y2 * y2;
        if (t2 >= 0) {
            t2 *= t2;
            n2 = t2 * t2 * grad(this.perm[ii + 1 + this.perm[jj + 1]], x2, y2);
        }

        return 70 * (n0 + n1 + n2);
    }

    // Fractional Brownian Motion for more organic noise
    fbm(x, y, octaves = 4) {
        let value = 0;
        let amplitude = 1;
        let frequency = 1;
        let maxValue = 0;

        for (let i = 0; i < octaves; i++) {
            value += amplitude * this.noise2D(x * frequency, y * frequency);
            maxValue += amplitude;
            amplitude *= 0.5;
            frequency *= 2;
        }

        return value / maxValue;
    }
}

// ============================================================================
// STATE
// ============================================================================

let canvas, ctx;
let width, height, dpr;
let data = { views: {}, points: [], links: {} };
let currentView = 'combined';
const VIEW_ORDER = ['combined', 'q1_safe_place', 'q2_stress', 'q3_understood', 'q4_free_day', 'q5_one_word'];
let hoveredPoint = null;
let selectedPoint = null;  // Click to lock selection
let hoveredNeighbors = new Set();
let hoveredLinks = new Set();
let time = 0;
let noise = new SimplexNoise(42);
let animationId = null;

// Background stars
let bgStars = [];

// Precomputed values
let pointsWithScreen = [];
let linkMap = new Map();
let clusterCenters = [];

// Smooth transitions
let targetBrightness = new Map();
let currentBrightness = new Map();

// Animation phases (preserved across resize)
let animationPhases = new Map();

// Smooth position transitions for view changes
let targetPositions = new Map();  // {index: {x, y}}
let currentPositions = new Map(); // {index: {x, y}}
let isTransitioning = false;

// DOM elements
let loading, infoPanel, pointCount, viewTabs, viewDescription, tabTooltip;
let panelNickname, panelCluster, panelContent, panelMeta;

// ============================================================================
// INITIALIZATION
// ============================================================================

async function init() {
    console.log('Initializing Hidden Connections...');

    canvas = document.getElementById('nebula');
    ctx = canvas.getContext('2d');

    loading = document.getElementById('loading');
    infoPanel = document.getElementById('info-panel');
    pointCount = document.getElementById('point-count');
    viewTabs = document.getElementById('view-tabs');
    viewDescription = document.getElementById('view-description');
    tabTooltip = document.getElementById('tab-tooltip');
    panelNickname = document.getElementById('panel-nickname');
    panelCluster = document.getElementById('panel-cluster');
    panelContent = document.getElementById('panel-content');
    panelMeta = document.getElementById('panel-meta');

    setupCanvas();
    generateBackgroundStars();
    console.log(`Canvas size: ${width}x${height}, DPR: ${dpr}`);

    window.addEventListener('resize', handleResize);
    window.addEventListener('keydown', handleKeyDown);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseleave', handleMouseLeave);
    canvas.addEventListener('click', handleClick);

    await loadData();

    loading.classList.add('hidden');

    console.log('Starting animation...');
    animate();
}

function setupCanvas() {
    dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    width = rect.width || window.innerWidth;
    height = rect.height || window.innerHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
}

function generateBackgroundStars() {
    bgStars = [];
    for (let i = 0; i < CONFIG.bgStars.count; i++) {
        bgStars.push({
            x: Math.random() * width,
            y: Math.random() * height,
            size: CONFIG.bgStars.minSize + Math.random() * (CONFIG.bgStars.maxSize - CONFIG.bgStars.minSize),
            phase: Math.random() * Math.PI * 2,
            speed: 0.5 + Math.random() * 1,
        });
    }
}

function handleResize() {
    setupCanvas();
    generateBackgroundStars();
    if (data.points.length > 0) {
        // Update target positions for new screen size
        setTargetPositions();
        // Jump directly to new positions (no animation on resize)
        for (const [index, target] of targetPositions) {
            currentPositions.set(index, { x: target.x, y: target.y });
        }
        computeScreenPositions();
        computeClusterCenters();
    }
}

// ============================================================================
// DATA LOADING
// ============================================================================

async function loadData() {
    try {
        const response = await fetch('points.json');
        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status}`);
        }
        const json = await response.json();

        data.views = json.views || {};
        data.points = json.points || [];
        data.links = json.links || {};

        console.log(`Loaded ${data.points.length} points with ${Object.keys(data.views).length} views`);

        pointCount.textContent = `${data.points.length} souls`;

        // Build the tabs UI
        buildViewTabs();

        // Initialize with combined view
        currentView = 'combined';
        computeScreenPositions();
        buildLinkMap();
        computeClusterCenters();
        initializeBrightness();
        initializePositions();
        updateViewDescription();

        console.log(`Computed ${clusterCenters.length} cluster centers`);

    } catch (error) {
        console.error('Failed to load data:', error);
        pointCount.textContent = 'Error loading data';
    }
}

function buildViewTabs() {
    viewTabs.innerHTML = '';

    for (const viewId of VIEW_ORDER) {
        const viewConfig = data.views[viewId];
        if (!viewConfig) continue;

        const tab = document.createElement('button');
        tab.className = 'view-tab' + (viewId === currentView ? ' active' : '');
        tab.textContent = viewConfig.shortLabel;
        tab.dataset.view = viewId;
        tab.dataset.description = viewConfig.description;

        tab.addEventListener('click', () => switchView(viewId));
        tab.addEventListener('mouseenter', showTabTooltip);
        tab.addEventListener('mouseleave', hideTabTooltip);

        viewTabs.appendChild(tab);
    }
}

function showTabTooltip(e) {
    const tab = e.currentTarget;
    const description = tab.dataset.description;
    if (!description || !tabTooltip) return;

    tabTooltip.textContent = description;

    // Position tooltip below the tab
    const rect = tab.getBoundingClientRect();
    const tabCenter = rect.left + rect.width / 2;

    // Calculate initial centered position
    let left = tabCenter - tabTooltip.offsetWidth / 2;
    const top = rect.bottom + 8;

    // Keep tooltip within viewport
    const padding = 12;
    if (left < padding) left = padding;
    if (left + tabTooltip.offsetWidth > window.innerWidth - padding) {
        left = window.innerWidth - tabTooltip.offsetWidth - padding;
    }

    // Calculate arrow offset to point at tab center
    const arrowOffset = tabCenter - left - tabTooltip.offsetWidth / 2;

    tabTooltip.style.left = `${left}px`;
    tabTooltip.style.top = `${top}px`;
    tabTooltip.style.setProperty('--arrow-offset', `${arrowOffset}px`);
    tabTooltip.classList.add('visible');
}

function hideTabTooltip() {
    if (tabTooltip) {
        tabTooltip.classList.remove('visible');
    }
}

function switchView(viewId) {
    if (viewId === currentView) return;
    if (!data.views[viewId]) return;

    console.log(`Switching to view: ${viewId}`);

    // Update tab active state
    const tabs = viewTabs.querySelectorAll('.view-tab');
    tabs.forEach(tab => {
        tab.classList.toggle('active', tab.dataset.view === viewId);
    });

    // Clear selection when switching views
    selectedPoint = null;
    hoveredPoint = null;

    // Set target positions for the new view
    currentView = viewId;
    setTargetPositions();
    buildLinkMap();
    computeScreenPositions();  // Update cluster info for new view
    computeClusterCenters();
    isTransitioning = true;

    // Update view description
    updateViewDescription();

    // Update hover state and panel
    updateHoverState();
    updatePanel();
}

function updateViewDescription() {
    const viewConfig = data.views[currentView];
    if (viewConfig && viewDescription) {
        viewDescription.textContent = viewConfig.description;
    }
}

function setTargetPositions() {
    const padding = 120;
    const availableWidth = width - padding * 2;
    const availableHeight = height - padding * 2;

    for (let i = 0; i < data.points.length; i++) {
        const point = data.points[i];
        const viewData = point.views[currentView];

        if (viewData) {
            const screenX = padding + ((viewData.x + 1) / 2) * availableWidth;
            const screenY = padding + ((1 - viewData.y) / 2) * availableHeight;
            targetPositions.set(i, { x: screenX, y: screenY });
        }
    }
}

function initializePositions() {
    const padding = 120;
    const availableWidth = width - padding * 2;
    const availableHeight = height - padding * 2;

    for (let i = 0; i < data.points.length; i++) {
        const point = data.points[i];
        const viewData = point.views[currentView];

        if (viewData) {
            const screenX = padding + ((viewData.x + 1) / 2) * availableWidth;
            const screenY = padding + ((1 - viewData.y) / 2) * availableHeight;
            currentPositions.set(i, { x: screenX, y: screenY });
            targetPositions.set(i, { x: screenX, y: screenY });
        }
    }
}

function computeScreenPositions() {
    const padding = 120;
    const availableWidth = width - padding * 2;
    const availableHeight = height - padding * 2;

    pointsWithScreen = data.points.map((point, index) => {
        // Preserve animation phases across resize - only generate once per point
        if (!animationPhases.has(point.id)) {
            animationPhases.set(point.id, {
                breathPhase: Math.random() * Math.PI * 2,
                twinklePhase: Math.random() * Math.PI * 2,
            });
        }
        const phases = animationPhases.get(point.id);

        // Get view-specific data
        const viewData = point.views[currentView] || point.views['combined'];

        // Use current animated positions if available, else calculate from view data
        let screenX, screenY;
        if (currentPositions.has(index)) {
            const pos = currentPositions.get(index);
            screenX = pos.x;
            screenY = pos.y;
        } else {
            screenX = padding + ((viewData.x + 1) / 2) * availableWidth;
            screenY = padding + ((1 - viewData.y) / 2) * availableHeight;
        }

        return {
            ...point,
            index,
            screenX,
            screenY,
            cluster: viewData.cluster,
            text: viewData.text,
            breathPhase: phases.breathPhase,
            twinklePhase: phases.twinklePhase,
        };
    });
}

function buildLinkMap() {
    linkMap.clear();

    // Get links for current view
    const viewLinks = data.links[currentView] || [];

    for (const link of viewLinks) {
        const [a, b] = link;

        if (!linkMap.has(a)) linkMap.set(a, new Set());
        if (!linkMap.has(b)) linkMap.set(b, new Set());

        linkMap.get(a).add(b);
        linkMap.get(b).add(a);
    }
}

function computeClusterCenters() {
    const clusterData = {};

    for (const point of pointsWithScreen) {
        if (!clusterData[point.cluster]) {
            clusterData[point.cluster] = { x: 0, y: 0, count: 0, points: [] };
        }
        clusterData[point.cluster].x += point.screenX;
        clusterData[point.cluster].y += point.screenY;
        clusterData[point.cluster].count++;
        clusterData[point.cluster].points.push(point);
    }

    clusterCenters = [];
    for (const [id, info] of Object.entries(clusterData)) {
        // Calculate spread for more accurate radius
        const cx = info.x / info.count;
        const cy = info.y / info.count;
        let maxDist = 0;
        for (const p of info.points) {
            const dist = Math.sqrt((p.screenX - cx) ** 2 + (p.screenY - cy) ** 2);
            if (dist > maxDist) maxDist = dist;
        }

        clusterCenters.push({
            cluster: parseInt(id),
            x: cx,
            y: cy,
            radius: Math.max(maxDist + 50, 80),
            count: info.count,
        });
    }
}

function initializeBrightness() {
    for (const point of pointsWithScreen) {
        targetBrightness.set(point.index, 0.5);
        currentBrightness.set(point.index, 0.5);
    }
}

// ============================================================================
// ANIMATION LOOP
// ============================================================================

function animate() {
    time += 1;

    // Smooth brightness transitions
    updateBrightness();

    // Smooth position transitions for view changes
    if (isTransitioning) {
        updatePositions();
    }

    // Clear with deep space gradient
    drawBackground();

    // Draw background stars
    drawBackgroundStars();

    // Draw nebula clouds
    drawNebulaClouds();

    // Draw constellation links
    drawLinks();

    // Draw participant stars
    drawStars();

    animationId = requestAnimationFrame(animate);
}

function updatePositions() {
    const transitionSpeed = 0.08;
    let stillTransitioning = false;

    for (const [index, target] of targetPositions) {
        const current = currentPositions.get(index);
        if (!current) continue;

        const dx = target.x - current.x;
        const dy = target.y - current.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist > 0.5) {
            stillTransitioning = true;
            current.x += dx * transitionSpeed;
            current.y += dy * transitionSpeed;
            currentPositions.set(index, current);
        } else {
            currentPositions.set(index, { x: target.x, y: target.y });
        }
    }

    // Update screen positions with interpolated values
    if (stillTransitioning) {
        computeScreenPositions();
        computeClusterCenters();
    } else {
        isTransitioning = false;
    }
}

function updateBrightness() {
    for (const point of pointsWithScreen) {
        const current = currentBrightness.get(point.index);
        const target = targetBrightness.get(point.index);
        const newValue = current + (target - current) * CONFIG.transitionSpeed;
        currentBrightness.set(point.index, newValue);
    }
}

// ============================================================================
// BACKGROUND
// ============================================================================

function drawBackground() {
    // Create a subtle gradient from deep blue-black to pure black
    const gradient = ctx.createRadialGradient(
        width / 2, height / 2, 0,
        width / 2, height / 2, Math.max(width, height) * 0.7
    );
    gradient.addColorStop(0, '#050510');
    gradient.addColorStop(0.5, '#030308');
    gradient.addColorStop(1, '#010103');

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
}

function drawBackgroundStars() {
    const t = time * CONFIG.bgStars.twinkleSpeed;

    for (const star of bgStars) {
        const twinkle = Math.sin(t * star.speed + star.phase) * 0.5 + 0.5;
        const alpha = 0.2 + twinkle * 0.4;

        ctx.beginPath();
        ctx.arc(star.x, star.y, star.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(200, 210, 255, ${alpha})`;
        ctx.fill();
    }
}

// ============================================================================
// NEBULA CLOUDS
// ============================================================================

function drawNebulaClouds() {
    const t = time * CONFIG.nebula.noiseSpeed;

    for (const center of clusterCenters) {
        const color = CONFIG.colors[center.cluster % CONFIG.colors.length];
        const baseRadius = center.radius;

        // Multiple layers with different noise offsets for depth
        for (let layer = 0; layer < CONFIG.nebula.layers; layer++) {
            const layerOffset = layer * 0.5;
            const layerDepth = 1 - (layer / CONFIG.nebula.layers);

            // Use FBM for organic-looking noise
            const noiseVal = noise.fbm(
                center.x * CONFIG.nebula.noiseScale + t,
                center.y * CONFIG.nebula.noiseScale + t * 0.7 + layerOffset,
                3
            );

            const distortX = noiseVal * 40 * CONFIG.nebula.drift;
            const distortY = noise.fbm(
                center.y * CONFIG.nebula.noiseScale + t * 0.5,
                center.x * CONFIG.nebula.noiseScale + layerOffset,
                3
            ) * 30 * CONFIG.nebula.drift;

            const radius = baseRadius * (1.2 + layer * 0.3) + noiseVal * 20;
            const alpha = CONFIG.nebula.baseAlpha * layerDepth * (0.8 + noiseVal * 0.2);

            // Draw cloud layer
            const gradient = ctx.createRadialGradient(
                center.x + distortX,
                center.y + distortY,
                0,
                center.x + distortX * 0.3,
                center.y + distortY * 0.3,
                radius
            );

            gradient.addColorStop(0, `rgba(${color.r}, ${color.g}, ${color.b}, ${alpha * 1.2})`);
            gradient.addColorStop(0.3, `rgba(${color.r}, ${color.g}, ${color.b}, ${alpha * 0.6})`);
            gradient.addColorStop(0.6, `rgba(${color.r}, ${color.g}, ${color.b}, ${alpha * 0.2})`);
            gradient.addColorStop(1, `rgba(${color.r}, ${color.g}, ${color.b}, 0)`);

            ctx.beginPath();
            ctx.arc(center.x + distortX * 0.5, center.y + distortY * 0.5, radius, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();
        }
    }
}

// ============================================================================
// LINKS (Constellation lines)
// ============================================================================

function drawLinks() {
    const pulseTime = time * CONFIG.link.pulseSpeed;
    const travelTime = time * CONFIG.link.travelSpeed;

    // Get links for current view
    const viewLinks = data.links[currentView] || [];

    for (const link of viewLinks) {
        const [aIdx, bIdx, strength] = link;
        const a = pointsWithScreen[aIdx];
        const b = pointsWithScreen[bIdx];

        if (!a || !b) continue;

        const linkKey = `${Math.min(aIdx, bIdx)}-${Math.max(aIdx, bIdx)}`;
        const isHovered = hoveredLinks.has(linkKey);

        // Get current brightness of connected points
        const brightA = currentBrightness.get(aIdx);
        const brightB = currentBrightness.get(bIdx);
        const linkBright = Math.max(brightA, brightB);

        // Use strength to scale link visibility (stronger connections more visible)
        // strength ranges from ~0.3 to ~1.0, normalize to 0.5-1.0 range for visual effect
        const strengthFactor = 0.5 + (strength || 0.5) * 0.5;

        // Calculate line properties - modulated by connection strength
        const baseAlpha = isHovered ? CONFIG.link.hoverAlpha : CONFIG.link.baseAlpha * strengthFactor;
        const lineWidth = isHovered ? CONFIG.link.hoverWidth : CONFIG.link.baseWidth * (0.7 + strengthFactor * 0.6);

        // Pulse effect - subtle wave along line
        const pulsePhase = (a.screenX + a.screenY) * 0.005;
        const pulse = Math.sin(pulseTime + pulsePhase) * 0.5 + 0.5;
        const alpha = baseAlpha * (0.6 + pulse * 0.4) * (0.5 + linkBright);

        // Get color from cluster
        const color = CONFIG.colors[a.cluster % CONFIG.colors.length];

        // Draw the base line
        ctx.beginPath();
        ctx.moveTo(a.screenX, a.screenY);
        ctx.lineTo(b.screenX, b.screenY);
        ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${alpha})`;
        ctx.lineWidth = lineWidth;
        ctx.lineCap = 'round';
        ctx.stroke();

        // Draw traveling particles on hovered links
        if (isHovered) {
            const dx = b.screenX - a.screenX;
            const dy = b.screenY - a.screenY;

            for (let i = 0; i < CONFIG.link.particleCount; i++) {
                const offset = i / CONFIG.link.particleCount;
                const pos = (travelTime * 2 + offset + pulsePhase) % 1;

                const px = a.screenX + dx * pos;
                const py = a.screenY + dy * pos;

                // Particle glow
                const particleGrad = ctx.createRadialGradient(px, py, 0, px, py, 6);
                particleGrad.addColorStop(0, `rgba(255, 255, 255, 0.9)`);
                particleGrad.addColorStop(0.3, `rgba(${color.r}, ${color.g}, ${color.b}, 0.6)`);
                particleGrad.addColorStop(1, `rgba(${color.r}, ${color.g}, ${color.b}, 0)`);

                ctx.beginPath();
                ctx.arc(px, py, 6, 0, Math.PI * 2);
                ctx.fillStyle = particleGrad;
                ctx.fill();

                // Core particle
                ctx.beginPath();
                ctx.arc(px, py, 2, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
                ctx.fill();
            }
        }
    }
}

// ============================================================================
// STARS
// ============================================================================

function drawStars() {
    const breathTime = time * CONFIG.star.breathSpeed;
    const twinkleTime = time * CONFIG.star.twinkleSpeed;

    // Sort by brightness so highlighted stars draw on top
    const sortedPoints = [...pointsWithScreen].sort((a, b) => {
        return currentBrightness.get(a.index) - currentBrightness.get(b.index);
    });

    // Calculate active point once outside the loop (optimization)
    const activePoint = selectedPoint || hoveredPoint;
    const activeIndex = activePoint ? activePoint.index : -1;

    for (const point of sortedPoints) {
        const brightness = currentBrightness.get(point.index);
        const isActive = point.index === activeIndex;
        const isNeighbor = hoveredNeighbors.has(point.index);

        // Breathing effect - slow, organic pulsation
        const breath = Math.sin(breathTime + point.breathPhase) * CONFIG.star.breathAmount;

        // Twinkle effect - faster, subtle shimmer
        const twinkle = Math.sin(twinkleTime + point.twinklePhase) * CONFIG.star.twinkleAmount;

        // Calculate radius
        let baseRadius = CONFIG.star.baseRadius;
        if (isActive) baseRadius = CONFIG.star.hoverRadius;
        else if (isNeighbor) baseRadius = CONFIG.star.neighborRadius;

        const radius = baseRadius * (1 + breath + twinkle * 0.5);

        const color = CONFIG.colors[point.cluster % CONFIG.colors.length];

        // Outer glow - larger, softer
        const glowRadius = CONFIG.star.glowRadius * (0.8 + brightness * 0.8);
        const glowAlpha = 0.1 + brightness * 0.35;

        const glowGradient = ctx.createRadialGradient(
            point.screenX, point.screenY, 0,
            point.screenX, point.screenY, glowRadius
        );
        glowGradient.addColorStop(0, `rgba(${color.r}, ${color.g}, ${color.b}, ${glowAlpha})`);
        glowGradient.addColorStop(0.3, `rgba(${color.r}, ${color.g}, ${color.b}, ${glowAlpha * 0.4})`);
        glowGradient.addColorStop(0.6, `rgba(${color.r}, ${color.g}, ${color.b}, ${glowAlpha * 0.1})`);
        glowGradient.addColorStop(1, `rgba(${color.r}, ${color.g}, ${color.b}, 0)`);

        ctx.beginPath();
        ctx.arc(point.screenX, point.screenY, glowRadius, 0, Math.PI * 2);
        ctx.fillStyle = glowGradient;
        ctx.fill();

        // Star core with bright center
        const coreGradient = ctx.createRadialGradient(
            point.screenX, point.screenY, 0,
            point.screenX, point.screenY, radius
        );

        // Brighter core for highlighted stars
        const coreWhite = isActive ? 255 : (isNeighbor ? 240 : 200 + brightness * 55);
        coreGradient.addColorStop(0, `rgba(${coreWhite}, ${coreWhite}, ${coreWhite}, ${0.8 + brightness * 0.2})`);
        coreGradient.addColorStop(0.4, `rgba(${Math.min(255, color.r + 60)}, ${Math.min(255, color.g + 60)}, ${Math.min(255, color.b + 60)}, ${0.7 + brightness * 0.3})`);
        coreGradient.addColorStop(1, `rgba(${color.r}, ${color.g}, ${color.b}, ${0.5 + brightness * 0.3})`);

        ctx.beginPath();
        ctx.arc(point.screenX, point.screenY, radius, 0, Math.PI * 2);
        ctx.fillStyle = coreGradient;
        ctx.fill();

        // Bright center point for highlighted stars
        if (brightness > 0.7) {
            ctx.beginPath();
            ctx.arc(point.screenX, point.screenY, radius * 0.3, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 255, 255, ${brightness * 0.9})`;
            ctx.fill();
        }
    }
}

// ============================================================================
// INTERACTION
// ============================================================================

function handleMouseMove(event) {
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    // Find nearest point
    let nearest = null;
    let nearestDist = CONFIG.hoverThreshold;

    for (const point of pointsWithScreen) {
        const dx = mouseX - point.screenX;
        const dy = mouseY - point.screenY;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < nearestDist) {
            nearestDist = dist;
            nearest = point;
        }
    }

    if (nearest !== hoveredPoint) {
        hoveredPoint = nearest;
        updateHoverState();
        updatePanel();
    }
}

function handleMouseLeave() {
    hoveredPoint = null;
    // Always update state - if selected, it will stay on selection
    // If not selected, everything will dim back to normal
    updateHoverState();
    updatePanel();
}

function handleClick(event) {
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    // Find nearest point
    let nearest = null;
    let nearestDist = CONFIG.hoverThreshold;

    for (const point of pointsWithScreen) {
        const dx = mouseX - point.screenX;
        const dy = mouseY - point.screenY;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < nearestDist) {
            nearestDist = dist;
            nearest = point;
        }
    }

    // Toggle selection
    if (nearest) {
        if (selectedPoint && selectedPoint.index === nearest.index) {
            // Clicking same point - deselect
            selectedPoint = null;
        } else {
            // Select new point
            selectedPoint = nearest;
        }
    } else {
        // Clicking empty space - deselect
        selectedPoint = null;
    }

    updateHoverState();
    updatePanel();
}

function handleKeyDown(event) {
    const currentIndex = VIEW_ORDER.indexOf(currentView);

    switch (event.key) {
        case 'ArrowLeft':
            // Previous view
            if (currentIndex > 0) {
                switchView(VIEW_ORDER[currentIndex - 1]);
            }
            event.preventDefault();
            break;

        case 'ArrowRight':
            // Next view
            if (currentIndex < VIEW_ORDER.length - 1) {
                switchView(VIEW_ORDER[currentIndex + 1]);
            }
            event.preventDefault();
            break;

        case 'Escape':
            // Clear selection
            if (selectedPoint) {
                selectedPoint = null;
                updateHoverState();
                updatePanel();
            }
            break;

        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
            // Quick jump to view by number
            const viewNum = parseInt(event.key) - 1;
            if (viewNum >= 0 && viewNum < VIEW_ORDER.length) {
                switchView(VIEW_ORDER[viewNum]);
            }
            break;
    }
}

// Navigate to a connected neighbor
function navigateToNeighbor(index) {
    const point = pointsWithScreen.find(p => p.index === index);
    if (point) {
        selectedPoint = point;
        hoveredPoint = point;
        updateHoverState();
        updatePanel();
    }
}

function updateHoverState() {
    hoveredNeighbors.clear();
    hoveredLinks.clear();

    // Reset all brightness targets
    for (const point of pointsWithScreen) {
        targetBrightness.set(point.index, 0.5);
    }

    // Use selected point if available, otherwise use hovered point
    const activePoint = selectedPoint || hoveredPoint;

    if (activePoint) {
        // Brighten active point
        targetBrightness.set(activePoint.index, 1.0);

        // Get and brighten neighbors
        const neighbors = linkMap.get(activePoint.index);
        if (neighbors) {
            for (const n of neighbors) {
                hoveredNeighbors.add(n);
                targetBrightness.set(n, 0.85);

                const linkKey = `${Math.min(activePoint.index, n)}-${Math.max(activePoint.index, n)}`;
                hoveredLinks.add(linkKey);
            }
        }

        // Dim non-connected points slightly
        for (const point of pointsWithScreen) {
            if (point.index !== activePoint.index && !hoveredNeighbors.has(point.index)) {
                targetBrightness.set(point.index, 0.3);
            }
        }
    }
}

function updatePanel() {
    // Use selected point if available, otherwise use hovered point
    const activePoint = selectedPoint || hoveredPoint;

    if (!activePoint) {
        infoPanel.classList.remove('visible');
        return;
    }

    infoPanel.classList.add('visible');

    // Nickname with fallback, add lock indicator if selected
    const nickname = activePoint.nickname || 'anonymous';
    const lockIndicator = selectedPoint ? ' \u2022' : '';  // bullet indicates locked
    panelNickname.textContent = nickname + lockIndicator;

    // Cluster badge with color (use current view's cluster)
    const color = CONFIG.colors[activePoint.cluster % CONFIG.colors.length];
    panelCluster.textContent = color.name || `Cluster ${activePoint.cluster + 1}`;
    panelCluster.style.background = `rgba(${color.r}, ${color.g}, ${color.b}, 0.15)`;
    panelCluster.style.color = `rgb(${color.r}, ${color.g}, ${color.b})`;
    panelCluster.style.borderColor = `rgba(${color.r}, ${color.g}, ${color.b}, 0.3)`;

    let html = '';

    // Get raw responses for this point
    const responses = activePoint.responses || {};

    const labelMap = {
        'q1_safe_place': 'Safe Place',
        'q2_stress': 'Stress Response',
        'q3_understood': 'Feeling Understood',
        'q4_free_day': 'Free Day',
        'q5_one_word': 'One Word',
    };

    // Determine which questions to show based on current view
    const viewConfig = data.views[currentView];
    const questionOrder = ['q1_safe_place', 'q2_stress', 'q3_understood', 'q4_free_day', 'q5_one_word'];

    if (currentView === 'combined') {
        // Show all responses
        for (const field of questionOrder) {
            const text = responses[field];
            if (text) {
                html += `
                    <div class="response-block">
                        <div class="response-label">${escapeHtml(labelMap[field])}</div>
                        <div class="response-text">${escapeHtml(text)}</div>
                    </div>
                `;
            }
        }
    } else {
        // Individual question view - show the featured question prominently
        const featuredField = currentView;
        const featuredText = responses[featuredField];

        if (featuredText) {
            html += `
                <div class="response-block featured">
                    <div class="response-label">${escapeHtml(viewConfig?.label || labelMap[featuredField])}</div>
                    <div class="response-text">${escapeHtml(featuredText)}</div>
                </div>
            `;
        }

        // Show other responses in a collapsed/secondary section
        const otherResponses = questionOrder.filter(f => f !== featuredField);
        let hasOthers = false;
        let othersHtml = '';

        for (const field of otherResponses) {
            const text = responses[field];
            if (text) {
                hasOthers = true;
                othersHtml += `
                    <div class="response-block secondary">
                        <div class="response-label">${escapeHtml(labelMap[field])}</div>
                        <div class="response-text">${escapeHtml(text)}</div>
                    </div>
                `;
            }
        }

        if (hasOthers) {
            html += `<div class="other-responses">${othersHtml}</div>`;
        }
    }

    // Add connected neighbors section
    const neighbors = linkMap.get(activePoint.index);
    if (neighbors && neighbors.size > 0) {
        html += `<div class="response-block neighbors-block">
            <div class="response-label">Connected Souls</div>
            <div class="neighbors-list">`;

        for (const neighborIdx of neighbors) {
            const neighbor = pointsWithScreen[neighborIdx];
            if (neighbor) {
                const neighborColor = CONFIG.colors[neighbor.cluster % CONFIG.colors.length];
                html += `
                    <button class="neighbor-link" onclick="navigateToNeighbor(${neighborIdx})"
                            style="border-color: rgba(${neighborColor.r}, ${neighborColor.g}, ${neighborColor.b}, 0.4)">
                        ${escapeHtml(neighbor.nickname || 'anonymous')}
                    </button>`;
            }
        }

        html += `</div></div>`;
    }

    panelContent.innerHTML = html;

    // Meta info - useful connection data
    const connections = neighbors?.size || 0;
    const clusterInfo = clusterCenters.find(c => c.cluster === activePoint.cluster);
    const clusterSize = clusterInfo?.count || 1;

    panelMeta.innerHTML = `
        <div class="meta-item">
            <span class="meta-label">Connections</span>
            <span class="meta-value">${connections} soul${connections !== 1 ? 's' : ''}</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Cluster</span>
            <span class="meta-value">${clusterSize} member${clusterSize !== 1 ? 's' : ''}</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">ID</span>
            <span class="meta-value">${escapeHtml(activePoint.id)}</span>
        </div>
    `;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================================
// START
// ============================================================================

document.addEventListener('DOMContentLoaded', init);
