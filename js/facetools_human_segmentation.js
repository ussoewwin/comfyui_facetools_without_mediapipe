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
const origProps = {};
const toggleWidget = (node, widget, show = false, suffix = "") => {
    if (!widget) return;
    if (!origProps[widget.name]) {
        origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };
    }
    const origSize = node.size;

    widget.type = show ? origProps[widget.name].origType : "facetoolsHidden" + suffix;
    widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];

    widget.linkedWidgets?.forEach(w => toggleWidget(node, w, show, ":" + widget.name));

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

function getTagList(tagArray) {
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
                    const updateWidgetValue = () => {
                        // Find the parent node and selector widget
                        let currentNode = el;
                        while (currentNode && !currentNode.comfyClass) {
                            currentNode = currentNode.parentElement;
                        }
                        if (currentNode && currentNode.widgets) {
                            const maskWidget = currentNode.widgets.find(w => w.name === 'mask_components');
                            if (maskWidget && maskWidget.value !== undefined) {
                                // Trigger value update by reading it (which will call the getter)
                                const currentValue = maskWidget.value;
                                // Force update by setting dirty flag
                                if (currentNode.graph) {
                                    currentNode.graph._version = (currentNode.graph._version || 0) + 1;
                                }
                                if (currentNode.setDirtyCanvas) {
                                    currentNode.setDirtyCanvas(true);
                                }
                            }
                        }
                    };
                    el.children[0].onclick = () => {
                        el.classList.toggle("facetools-prompt-styles-tag-selected");
                        updateSelectedStyle();
                        // Update widget value to trigger save
                        updateWidgetValue();
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
                let method_values = '';

                this.setProperty("values", []);

                // Check if addDOMWidget exists
                if (!this.addDOMWidget) {
                    console.warn('[facetools] addDOMWidget not available');
                    return;
                }

                console.log('[facetools] Creating DOM widget for mask_components');
                
                // Remove existing mask_components widget if it exists
                const existingMaskWidget = this.widgets.find((w) => w.name === 'mask_components');
                if (existingMaskWidget) {
                    console.log('[facetools] Removing existing mask_components widget');
                    const index = this.widgets.indexOf(existingMaskWidget);
                    if (index !== -1) {
                        this.widgets.splice(index, 1);
                    }
                }
                
                // Create DOM widget with inline styles to ensure CSS is applied
                const containerDiv = $el('div.facetools-prompt-styles', {
                    style: {
                        overflow: 'auto',
                        width: '100%'
                    }
                }, [list]);
                let selector = this.addDOMWidget('mask_components', "btn", containerDiv);
                console.log('[facetools] DOM widget created:', selector);
                console.log('[facetools] selector.element:', selector?.element);
                console.log('[facetools] selector.element type:', selector?.element?.constructor?.name);

                    // Store original value getter/setter if they exist
                    const methodWidget = this.widgets[method];
                    if (methodWidget) {
                        // Check if property descriptor already exists
                        const descriptor = Object.getOwnPropertyDescriptor(methodWidget, 'value');
                        if (descriptor && !descriptor.configurable) {
                            // If not configurable, we can't redefine it - use a different approach
                            console.warn('[facetools] Cannot redefine method widget value property');
                            return;
                        }
                        
                        Object.defineProperty(methodWidget, 'value', {
                            configurable: true,
                            enumerable: true,
                            set: (value) => {
                                method_values = value;
                                if (method_values && selector && selector.element && selector.element.children && selector.element.children[0]) {
                                    selector.element.children[0].innerHTML = '';
                                    if (method_values == 'selfie_multiclass_256x256') {
                                        toggleWidget(this, findWidgetByName(this, 'confidence'), true);
                                        this.setSize([300, 260]);
                                    } else {
                                        toggleWidget(this, findWidgetByName(this, 'confidence'));
                                        this.setSize([300, 500]);
                                    }
                                    if (tags[method_values]) {
                                        let list = getTagList(tags[method_values]);
                                        selector.element.children[0].append(...list);
                                    }
                                }
                            },
                            get: () => {
                                return method_values;
                            }
                        });
                    }

                let mask_select_values = '';

                // Define selector.value property
                // Store a reference to the node for use in getter
                const nodeRef = this;
                
                // First, check if value property already exists
                const existingDescriptor = Object.getOwnPropertyDescriptor(selector, 'value');
                if (existingDescriptor && !existingDescriptor.configurable) {
                    console.warn('[facetools] selector.value property exists and is not configurable, will use wrapper approach');
                    // If not configurable, we'll wrap the existing property
                    const originalGetter = existingDescriptor.get;
                    const originalSetter = existingDescriptor.set;
                    const originalValue = existingDescriptor.value;
                    
                    // Try to override by setting a new property on the prototype or using a different approach
                    // For now, we'll just log a warning and continue
                }
                
                try {
                    // Try to delete existing property if configurable
                    if (existingDescriptor && existingDescriptor.configurable) {
                        delete selector.value;
                    }
                    
                    Object.defineProperty(selector, "value", {
                        configurable: true,
                        enumerable: true,
                        set: function(value) {
                            // Store the value for persistence
                            this._internalValue = value || '';
                            mask_select_values = value || '';
                            setTimeout(() => {
                                if (this.element && this.element.children && this.element.children[0]) {
                                    const tagList = this.element.children[0];
                                    if (tagList) {
                                        tagList.querySelectorAll(".facetools-prompt-styles-tag").forEach(el => {
                                            let arr = (value || '').split(',').filter(v => v.trim() !== '');
                                            if (arr.includes(el.dataset.tag)) {
                                                el.classList.add("facetools-prompt-styles-tag-selected");
                                                if (el.children && el.children[0]) {
                                                    el.children[0].checked = true;
                                                }
                                                // Update style
                                                el.style.backgroundColor = 'var(--theme-color-light)';
                                                el.style.borderColor = 'var(--theme-color-light)';
                                                el.style.color = 'var(--comfy-menu-bg)';
                                            } else {
                                                el.classList.remove("facetools-prompt-styles-tag-selected");
                                                if (el.children && el.children[0]) {
                                                    el.children[0].checked = false;
                                                }
                                                // Update style
                                                el.style.backgroundColor = 'var(--comfy-input-bg)';
                                                el.style.borderColor = 'var(--border-color)';
                                                el.style.color = 'var(--input-text)';
                                            }
                                        });
                                        // Update node properties
                                        if (nodeRef && nodeRef.properties) {
                                            const selectedValues = (value || '').split(',').filter(v => v.trim() !== '');
                                            nodeRef.properties["values"] = selectedValues;
                                        }
                                    }
                                }
                            }, 100);
                        },
                        get: function() {
                            if (this.element && this.element.children && this.element.children[0]) {
                                const tagList = this.element.children[0];
                                if (tagList) {
                                    const selectedValues = [];
                                    tagList.querySelectorAll(".facetools-prompt-styles-tag").forEach(el => {
                                        if (el.classList.contains("facetools-prompt-styles-tag-selected")) {
                                            selectedValues.push(el.dataset.tag);
                                        }
                                    });
                                    mask_select_values = selectedValues.join(',');
                                    // Also update node properties for persistence
                                    if (nodeRef && nodeRef.properties) {
                                        nodeRef.properties["values"] = selectedValues;
                                    }
                                    // Store in widget's internal value for ComfyUI to save
                                    if (this._internalValue !== mask_select_values) {
                                        this._internalValue = mask_select_values;
                                    }
                                    return mask_select_values || '';
                                }
                            }
                            // Return stored value if available (for restoration)
                            if (this._internalValue !== undefined) {
                                return this._internalValue;
                            }
                            return mask_select_values || '';
                        }
                    });
                } catch (e) {
                    console.warn('[facetools] Could not define selector.value property:', e);
                    // Fallback: try to use the existing value property if it exists
                    if (selector.value !== undefined) {
                        console.log('[facetools] Using existing selector.value property');
                    }
                }

                // 初始化
                setTimeout(_ => {
                    console.log('[facetools] Initializing widget, method_values:', method_values);
                    if (!method_values) {
                        method_values = 'selfie_multiclass_256x256';
                    }
                    console.log('[facetools] selector:', selector);
                    console.log('[facetools] selector.element:', selector?.element);
                    
                    // Try to find the ul element in different ways
                    let tagListContainer = null;
                    if (selector) {
                        // Case 1: selector.element is the div, and children[0] is the ul
                        if (selector.element && selector.element.children && selector.element.children[0]) {
                            tagListContainer = selector.element.children[0];
                            console.log('[facetools] Found ul via selector.element.children[0]');
                        }
                        // Case 2: selector.element is directly the div, and we need to find ul inside
                        else if (selector.element) {
                            const ulElement = selector.element.querySelector('ul.facetools-prompt-styles-list');
                            if (ulElement) {
                                tagListContainer = ulElement;
                                console.log('[facetools] Found ul via querySelector');
                            }
                        }
                        // Case 3: selector itself might be the element
                        else if (selector.querySelector) {
                            const ulElement = selector.querySelector('ul.facetools-prompt-styles-list');
                            if (ulElement) {
                                tagListContainer = ulElement;
                                console.log('[facetools] Found ul via selector.querySelector');
                            }
                        }
                    }
                    
                    if (tagListContainer) {
                        console.log('[facetools] Setting up tag list for method:', method_values);
                        tagListContainer.innerHTML = '';
                        if (tags[method_values]) {
                            let list = getTagList(tags[method_values]);
                            tagListContainer.append(...list);
                            console.log('[facetools] Tag list appended, count:', list.length);
                        }
                    } else {
                        console.warn('[facetools] Could not find tag list container');
                        console.warn('[facetools] selector keys:', selector ? Object.keys(selector) : 'null');
                        console.warn('[facetools] selector.element:', selector?.element);
                        console.warn('[facetools] selector.element type:', selector?.element?.constructor?.name);
                    }
                    
                    if (method_values == 'selfie_multiclass_256x256') {
                        toggleWidget(this, findWidgetByName(this, 'confidence'), true);
                        this.setSize([300, 260]);
                    } else {
                        toggleWidget(this, findWidgetByName(this, 'confidence'));
                        this.setSize([300, 500]);
                    }
                }, 1);

            } catch (e) {
                console.error('[facetools] Error in onNodeCreated:', e);
            }
        }
    }
});

