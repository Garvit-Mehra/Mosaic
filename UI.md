# Mosaic UI: Spatial & Liquid Design System

> **Vision**: The Mosaic UI is a premium, spatial, and fluid design system. Inspired by modern liquid glass aesthetics and spatial computing interfaces (like visionOS and macOS), it brings digital surfaces to life with "buttery smooth" animations, frosted translucent panels, inner glow, and deep, dynamic drop shadows.

This specification serves as the implementation guide for developers to translate the Mosaic design language into code.

---

## 1. Core Principles

- **Liquid Glass (Glassmorphism + Spatial)**: Surfaces aren't solid; they are translucent, frosted, and refractive. They reveal blurred, colorful organic shapes beneath them. They have subtle inner borders to mimic light catching the edge of a glass pane.
- **Buttery Smooth Dynamics**: Nothing snaps instantly. Every interaction, transition, and state change is interpolated with physics-based, spring-driven animations. Interactions feel tactile and bouncy.
- **Continuous Curves (Squircles)**: Avoid harsh edges. Use generous border radii, particularly continuous curves (squircles), soft gradients, and elements that feel like they belong to a cohesive, living ecosystem.
- **Depth & Hierarchy**: Use shadows, z-index, and varying levels of backdrop-blur to establish a clear spatial hierarchy. Backgrounds push back, while interactive elements pull forward.

---

## 2. Visual Foundation

### 2.1 Colors & Themes
Mosaic relies heavily on a vibrant, organic background layer overlaid with neutral, translucent UI panels.

- **Background Palette (The "Canvas")**: Use vibrant, fluid gradients (Mesh Gradients) that slowly morph over time.
  - *Aurora*: `#4facfe` to `#00f2fe`
  - *Sunset*: `#f093fb` to `#f5576c`
  - *Mint*: `#43e97b` to `#38f9d7`
- **Surface Materials**:
  - *Thick Material*: `rgba(255, 255, 255, 0.7)` (For headers, prominent cards)
  - *Thin Material*: `rgba(255, 255, 255, 0.3)` (For secondary panels, lists)
  - *Dark Mode Materials*: `rgba(30, 30, 30, 0.7)` / `rgba(30, 30, 30, 0.3)`
- **Text & Typography**:
  - *Primary Text*: `rgba(0, 0, 0, 0.9)` (Light) / `rgba(255, 255, 255, 0.9)` (Dark)
  - *Secondary Text*: `rgba(0, 0, 0, 0.55)` (Light) / `rgba(255, 255, 255, 0.55)` (Dark)

### 2.2 Liquid Glass (The Spatial Effect)
To achieve the signature Mosaic look, UI panels must combine background blur, semi-transparent backgrounds, and subtle lighting (inner rim light).

**CSS Implementation Details:**
```css
.mosaic-panel {
  /* The Glass */
  background: rgba(255, 255, 255, 0.4);
  backdrop-filter: blur(40px) saturate(200%);
  -webkit-backdrop-filter: blur(40px) saturate(200%); /* Safari support */
  
  /* The Edge Lighting (Inner Highlight) */
  border: 1px solid rgba(255, 255, 255, 0.5);
  box-shadow: 
    inset 0 1px 1px rgba(255, 255, 255, 0.8), /* Top rim light */
    0 10px 40px -10px rgba(0, 0, 0, 0.15); /* Soft drop shadow for elevation */
  
  /* Organic Shapes */
  border-radius: 32px; /* Emulate squircle if possible using mask-image or custom shapes */
}

/* Dark Mode Variations */
@media (prefers-color-scheme: dark) {
  .mosaic-panel {
    background: rgba(40, 40, 40, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 
      inset 0 1px 1px rgba(255, 255, 255, 0.15),
      0 10px 40px -10px rgba(0, 0, 0, 0.5);
  }
}
```

### 2.3 Typography
Typography should be modern, geometric, and highly legible, akin to Apple's SF Pro. 
- **Font Family**: Inter, SF Pro Display/Text, or Roboto.
- **Weight & Contrast**: Use bold (700) for large headers and medium (500) or regular (400) for body.
- **Tracking/Letter Spacing**: Tighten the tracking slightly on large headers (e.g., `-0.02em` or `-0.04em`) and loosen it on uppercase sub-labels (e.g., `0.05em`).

---

## 3. Motion & Animation (Buttery Smooth)

This is the most critical part of Mosaic. Interactions must feel physical, like manipulating a fluid or a well-oiled spring.

### 3.1 Spring Physics
Do not use standard CSS linear or ease-in-out transitions for structural elements. Use spring physics. If using React, Framer Motion or React Spring is highly recommended.

**Framer Motion Example:**
```javascript
const butterySpring = {
  type: "spring",
  stiffness: 400,
  damping: 30,
  mass: 1,
  restDelta: 0.001
};

const bouncySpring = {
  type: "spring",
  stiffness: 300,
  damping: 20,
  mass: 1
};
```

### 3.2 Micro-Interactions
- **Hover States**: Elements like cards and buttons should slightly scale up (`scale: 1.03`), and the drop shadow should become larger and softer to indicate the element floating closer to the user.
- **Click/Tap (Squish)**: Elements should feel tactile. On active state (mousedown/touchstart), elements should squish down notably (`scale: 0.95` or `0.92`), mimicking pressing a physical button.
- **Liquid Morphing (Shared Element Transitions)**: When a component expands (e.g., opening a card into a modal), the transition should smoothly morph the dimensions, border radii, and background color without any harsh snapping.

### 3.3 The "Breathe" Effect
Background gradient meshes should slowly shift and rotate in the background. It shouldn't be distracting, just enough to make the interface feel "alive".

---

## 4. Components

### 4.1 Buttons
Buttons should feel like polished, frosted glass pills.
- **Primary Button**: Uses an internal vibrant gradient (e.g., matching the Canvas) with a glass overlay and a distinct inner rim light. It pulses slightly on hover.
- **Secondary Button**: Standard glass panel (`rgba(255, 255, 255, 0.2)`) with a subtle white border. High background blur.
- **Shape**: Fully rounded (`border-radius: 9999px`) or heavily rounded rectangles (`24px`).

### 4.2 Cards
Cards should have generous padding (`min 24px`) and corner radii (`32px`). Avoid harsh borders inside the cards; use dividers made of very low-opacity white/black lines (`rgba(0,0,0,0.05)` or `rgba(255,255,255,0.1)`).

### 4.3 Inputs & Text Fields
Inputs shouldn't look like rigid boxes. They should look like soft indentations or frosted pills in the glass.
- **Resting State**: `rgba(0,0,0,0.03)` background, inset shadow.
- **Focus State**: The border shouldn't just change color; a soft, glowing aura should bloom around the input, and the background should become slightly more opaque. The transition into focus must use the `butterySpring`.
- **Selection**: Text selection highlight should match the vibrant accent color, heavily rounded.

### 4.4 Modals & Popovers
When a modal appears, the background content should scale down slightly (`scale: 0.95`) and blur deeply, creating a sense of depth and focus on the foreground modal. The modal itself should slide up with a spring animation and heavy drop shadow.

---

## 5. Developer Checklist for Implementation

- [ ] Implement a global mesh gradient background that slowly, fluidly animates.
- [ ] Create utility classes/components for the `MosaicPanel` (the core glass element with rim light).
- [ ] Set up a motion library (like Framer Motion) and define the default `butterySpring` and `bouncySpring` configurations.
- [ ] Ensure all hover, focus, and active (squish) states use spring animations rather than CSS `ease` or `linear`.
- [ ] Implement spatial depth: scale down and blur backgrounds when foreground overlays (modals) appear.
- [ ] Audit for accessibility: Ensure text has sufficient contrast against the blurred, colorful backgrounds. Use subtle text-shadows if necessary to maintain legibility on glass surfaces.
