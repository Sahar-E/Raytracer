//
// Created by Sahar on 22/07/2022.
//

#pragma once


class ImGuiLayerUtils {
public:

    /**
     * Start ImGui frame, which is called in the start of the main loop.
     */
    static void startImGuiFrame();

    /**
     * End ImGui, which is called in the end of the main loop.
     */
    static void endImGuiFrame();

    /**
     * Cleans up ImGui frame, when ImGui is destroyed.
     */
    static void imGuiCleanup();
};
