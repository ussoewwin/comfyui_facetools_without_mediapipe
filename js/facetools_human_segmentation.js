/**
 * Mask Components Selector Widget for Facetools Human Segmentation
 * Adapted from comfyui-easy-use's easySeg.js
 * 
 * This integration script is licensed under the GNU General Public License v3.0 (GPL-3.0).
 * Original source: https://github.com/lllyasviel/ComfyUI-Easy-Use
 */

import {app} from "/scripts/app.js";
import {$el} from "/scripts/ui.js";

// Load CSS file (deferred to avoid blocking)
const addCss = () => {
    try {
        // Check if CSS is already loaded
        const cssFileName = 'facetools_human_segmentation.css';
        const existingLink = document.querySelector(`link[href*="${cssFileName}"]`);
        if (existingLink) return;
        
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.type = "text/css";
        // ComfyUI extensions are served from /extensions/<extension_name>/
        // Since WEB_DIRECTORY is "./js", the path should be /extensions/comfyui_facetools_without_mediapipe/js/facetools_human_segmentation.css
        const extensionName = 'comfyui_facetools_without_mediapipe';
        link.href = `/extensions/${extensionName}/js/${cssFileName}`;
        link.onerror = () => {
            // Fallback: try without leading slash
            link.href = `extensions/${extensionName}/js/${cssFileName}`;
            link.onerror = () => console.warn(`[facetools] Failed to load CSS: ${cssFileName}`);
        };
        document.head.appendChild(link);
    } catch (e) {
        console.warn(`[facetools] Error loading CSS: ${e}`);
    }
};

// Load CSS asynchronously after DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        setTimeout(() => addCss(), 100);
    });
} else {
    setTimeout(() => addCss(), 100);
}

// Helper function to find widget by name
const findWidgetByName = (node, name) => node.widgets.find((w) => w.name === name);

// Helper function to toggle widget visibility
// NOTE: We must keep the original widget for workflow persistence, but fully disable drawing.
const origProps = {};
const toggleWidget = (node, widget, show = false) => {
    if (!widget) return;
    if (!origProps[widget.name]) {
        origProps[widget.name] = {
            origType: widget.type,
            origComputeSize: widget.computeSize,
            origDraw: widget.draw,
        };
    }
    const origSize = node.size;

    if (show) {
        widget.type = origProps[widget.name].origType;
        widget.computeSize = origProps[widget.name].origComputeSize;
        widget.draw = origProps[widget.name].origDraw;
    } else {
        // Use a standard hidden type and null-draw so ComfyUI doesn't render an input.
        widget.type = "hidden";
        widget.computeSize = () => [0, 0];
        widget.draw = () => {};
    }

    widget.linkedWidgets?.forEach(w => toggleWidget(node, w, show));

    const height = show ? Math.max(node.computeSize()[1], origSize[1]) : node.size[1];
    node.setSize([node.size[0], height]);
};

// Tag lists for each method
const tags = {
    "selfie_multiclass_256x256": ["Background", "Hair", "Body", "Face", "Clothes", "Others"],
    "human_parsing_lip": ["Background", "Hat", "Hair", "Glove", "Sunglasses", "Upper-clothes", "Dress", "Coat", "Socks", "Pants", "Jumpsuits", "Scarf", "Skirt", "Face", "Left-arm", "Right-arm", "Left-leg", "Right-leg", "Left-shoe", "Right-shoe"],
    "human_parts (deeplabv3p)": ["Background", "Hair", "Body", "Face", "Left-arm", "Right-arm", "Left-leg", "Right-leg"],
    "segformer_b3_clothes": ["Background", "Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm", "Bag", "Scarf"],
    "segformer_b3_fashion": ["Background", "Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm", "Bag", "Scarf"],
    "face_parsing": ["Background", "Skin", "Left-eyebrow", "Right-eyebrow", "Left-eye", "Right-eye", "Nose", "Mouth", "Upper-lip", "Lower-lip"]
};

function getTagList(tagArray, storeWidget, nodeRef) {
    let rlist = [];
    tagArray.forEach((k, i) => {
        rlist.push($el(
            "label.facetools-prompt-styles-tag",
            {
                dataset: {
                    tag: i,
                    name: k,
                    index: i
                },
                style: {
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '6px 10px',
                    color: 'var(--input-text)',
                    backgroundColor: 'var(--comfy-input-bg)',
                    borderRadius: '6px',
                    border: '2px solid var(--border-color)',
                    fontSize: '11px',
                    cursor: 'pointer',
                    whiteSpace: 'nowrap',
                    transition: 'all 0.2s ease'
                },
                $: (el) => {
                    const updateSelectedStyle = () => {
                        if (el.classList.contains('facetools-prompt-styles-tag-selected')) {
                            el.style.backgroundColor = 'var(--theme-color-light)';
                            el.style.borderColor = 'var(--theme-color-light)';
                            el.style.color = 'var(--comfy-menu-bg)';
                        } else {
                            el.style.backgroundColor = 'var(--comfy-input-bg)';
                            el.style.borderColor = 'var(--border-color)';
                            el.style.color = 'var(--input-text)';
                        }
                    };
                    // expose for external restore
                    el.__facetoolsUpdateSelectedStyle = updateSelectedStyle;
                    el.children[0].onclick = () => {
                        el.classList.toggle("facetools-prompt-styles-tag-selected");
                        updateSelectedStyle();
                        // Persist selection by updating the original ComfyUI widget value (this is what gets saved)
                        try {
                            const ul = el.closest('ul.facetools-prompt-styles-list');
                            const selected = [];
                            if (ul) {
                                ul.querySelectorAll('label.facetools-prompt-styles-tag').forEach(lbl => {
                                    if (lbl.classList.contains('facetools-prompt-styles-tag-selected')) {
                                        selected.push(lbl.dataset.tag);
                                    }
                                });
                            }
                            const val = selected.length ? selected.join(',') : '0';
                            if (storeWidget) {
                                storeWidget.value = val;
                                if (typeof storeWidget.callback === 'function') storeWidget.callback(val);
                            }
                            if (nodeRef?.graph) nodeRef.graph._version = (nodeRef.graph._version || 0) + 1;
                            if (nodeRef?.setDirtyCanvas) nodeRef.setDirtyCanvas(true, true);
                        } catch (e) {
                            console.warn('[facetools] Failed to persist mask_components selection:', e);
                        }
                    };
                    el.onmouseenter = () => {
                        if (!el.classList.contains('facetools-prompt-styles-tag-selected')) {
                            el.style.filter = 'brightness(1.15)';
                            el.style.borderColor = 'var(--theme-color-light)';
                        }
                    };
                    el.onmouseleave = () => {
                        if (!el.classList.contains('facetools-prompt-styles-tag-selected')) {
                            el.style.filter = '';
                            el.style.borderColor = 'var(--border-color)';
                        }
                    };
                    // Initialize style
                    updateSelectedStyle();
                },
            },
            [
                $el("input", {
                    type: 'checkbox',
                    name: i,
                    style: {
                        margin: '0',
                        width: '1rem',
                        height: '1rem'
                    }
                }),
                $el("span", {
                    textContent: k,
                    style: {
                        margin: '0',
                        lineHeight: '1.2'
                    }
                })
            ]
        ));
    });
    return rlist;
}

app.registerExtension({
    name: 'comfyui_facetools_without_mediapipe.human_segmentation',
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Match ONLY our specific node - be very strict to avoid affecting other nodes
        // nodeData.name is the key from NODE_CLASS_MAPPINGS: "facetools_humanSegmentationIF"
        console.log('[facetools] beforeRegisterNodeDef called for:', nodeData.name);
        if (nodeData.name !== 'facetools_humanSegmentationIF') {
            return; // Exit early if not our node
        }
        
        console.log('[facetools] Matched facetools_humanSegmentationIF node, setting up widget');
            
        // 创建时
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            try {
                // Double check node name at runtime to be absolutely sure
                const nodeName = this.type || this.comfyClass || '';
                console.log('[facetools] onNodeCreated called for node:', nodeName);
                if (nodeName !== 'facetools_humanSegmentationIF') {
                    // Not our node, call original and return
                    if (onNodeCreated) {
                        onNodeCreated.apply(this, arguments);
                    }
                    return;
                }
                
                console.log('[facetools] Processing facetools_humanSegmentationIF node');
                onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                // Use findIndex like the original code
                const method = this.widgets.findIndex((w) => w.name == 'method');
                if (method === -1) {
                    console.warn('[facetools] method widget not found');
                    return;
                }
                
                console.log('[facetools] Found method widget at index:', method);
                const list = $el("ul.facetools-prompt-styles-list.no-top", {
                    style: {
                        display: 'flex',
                        flexWrap: 'wrap',
                        gap: '6px',
                        listStyle: 'none',
                        padding: '8px',
                        margin: '0'
                    }
                }, []);

                const nodeRef = this;

                // Check if addDOMWidget exists
                if (!this.addDOMWidget) {
                    console.warn('[facetools] addDOMWidget not available');
                    return;
                }

                console.log('[facetools] Creating DOM widget for mask_components');

                // Keep the original widget (mask_components) as the true persisted value container
                let storeWidget = findWidgetByName(this, 'mask_components');
                if (!storeWidget && this.addWidget) {
                    // safety fallback
                    storeWidget = this.addWidget('text', 'mask_components', '0');
                }
                // Hide it, but do NOT remove it (ComfyUI saves this widget value in workflow)
                toggleWidget(this, storeWidget, false, '_store');
                
                // Create DOM widget with inline styles to ensure CSS is applied
                const containerDiv = $el('div.facetools-prompt-styles', {
                    style: {
                        overflow: 'auto',
                        width: '100%'
                    }
                }, [list]);
                // Create a separate UI widget (does not get serialized), name must NOT clash with input name
                const selectorUI = this.addDOMWidget('mask_components_ui', "btn", containerDiv);
                console.log('[facetools] DOM widget created:', selectorUI);

                const methodWidget = this.widgets[method];

                const applySelectionFromValue = (val) => {
                    const arr = (val || '0').split(',').map(v => v.trim()).filter(v => v !== '');
                    list.querySelectorAll('label.facetools-prompt-styles-tag').forEach(lbl => {
                        const selected = arr.includes(lbl.dataset.tag);
                        lbl.classList.toggle('facetools-prompt-styles-tag-selected', selected);
                        if (lbl.__facetoolsUpdateSelectedStyle) lbl.__facetoolsUpdateSelectedStyle();
                        const cb = lbl.querySelector('input[type=\"checkbox\"]');
                        if (cb) cb.checked = selected;
                    });
                };

                const rebuildTagList = (methodValue) => {
                    const m = methodValue || 'selfie_multiclass_256x256';
                    list.innerHTML = '';

                    if (m === 'selfie_multiclass_256x256') {
                        toggleWidget(nodeRef, findWidgetByName(nodeRef, 'confidence'), true);
                        nodeRef.setSize([300, 260]);
                    } else {
                        toggleWidget(nodeRef, findWidgetByName(nodeRef, 'confidence'));
                        nodeRef.setSize([300, 500]);
                    }

                    const tagArray = tags[m] || [];
                    const nodes = getTagList(tagArray, storeWidget, nodeRef);
                    list.append(...nodes);
                    // restore selection from the persisted widget value
                    applySelectionFromValue(storeWidget?.value || '0');
                };

                // 初始化
                setTimeout(_ => {
                    // initial render
                    rebuildTagList(methodWidget?.value);
                    // hook method changes (no Object.defineProperty: avoids conflicts)
                    if (methodWidget) {
                        const origCb = methodWidget.callback;
                        methodWidget.callback = (v) => {
                            if (origCb) origCb(v);
                            rebuildTagList(v);
                        };
                    }
                }, 1);

            } catch (e) {
                console.error('[facetools] Error in onNodeCreated:', e);
            }
        }
    }
});

