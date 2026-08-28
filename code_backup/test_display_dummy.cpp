/*
 * test_display_dummy.cpp
 *
 * Standalone smoke test for the CComposite / CEventCompositorSource display
 * path, WITHOUT going through CEventFrameRenderer or any real event data.
 *
 * Goal: isolate "does a source posting RGBA frames into CComposite actually
 * reach the screen" from "does the event renderer produce correct pixels".
 * Once this confirms the display pipeline (buffer alloc -> registration ->
 * Post -> WfdFlip) is working end to end, swap the synthetic pattern below
 * for real evsim::EventPacket -> CEventFrameRenderer output.
 *
 * It intentionally skips CNvSIPLMaster/camera pipeline setup entirely - the
 * compositor only needs an NvSciBufModule and NvSciSyncModule, which it can
 * get directly from NvSciBuf/NvSciSync, same as CNvSIPLMaster::Setup() does
 * under the hood.
 *
 * Also exercises the byte-order ASSUMPTION flagged in
 * CEventCompositorSource.hpp: draws a pure-red quadrant and a pure-blue
 * quadrant so a swapped R/B channel is immediately visible on screen,
 * matching the polarity colors CEventFrameRenderer::PlotEvent uses.
 *
 * NOTE: this depends on NVIDIA's proprietary NvSciBuf/NvSciSync/NvMedia/WFD
 * headers and libraries (and on CNvWfd.hpp / CCompositeHelper.hpp, which
 * weren't part of this file set), so it can only be built inside the DRIVE
 * SDK toolchain - it will not compile in this sandbox. Treat this as a
 * reviewable/buildable-on-target source file, not something I ran here.
 */

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <thread>
#include <vector>

#include "CComposite.hpp"
#include "CEventCompositorSource.hpp"
#include "CUtils.hpp"

#include "nvscibuf.h"
#include "nvscisync.h"

static std::atomic<bool> bQuit{false};

static void SigHandler(int)
{
    bQuit = true;
}

// Builds one RGBA8 test frame: black background, top-left quadrant pure
// red, bottom-right quadrant pure blue, with a thin white crosshair through
// the middle so scaling/rotation issues on the actual panel are also
// obvious. `phase` shifts the crosshair each frame just so it's visibly
// live rather than a static image (helps rule out a frozen/duplicated
// buffer bug in the round-robin pool).
static std::vector<uint8_t> MakeTestFrame(int width, int height, int phase)
{
    std::vector<uint8_t> frame(static_cast<size_t>(width) * height * 4, 0);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = (static_cast<size_t>(y) * width + x) * 4;
            uint8_t r = 0, g = 0, b = 0, a = 255;

            bool leftHalf = x < (width / 2);
            bool topHalf  = y < (height / 2);
            if (leftHalf && topHalf) {
                r = 255; // matches CEventFrameRenderer's ON-event red
            } else if (!leftHalf && !topHalf) {
                b = 255; // matches CEventFrameRenderer's OFF-event blue
            }

            int crossX = (width / 2 + phase) % width;
            int crossY = height / 2;
            if (x == crossX || y == crossY) {
                r = g = b = 255;
            }

            frame[idx + 0] = r;
            frame[idx + 1] = g;
            frame[idx + 2] = b;
            frame[idx + 3] = a;
        }
    }
    return frame;
}

int main()
{
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    const int width  = 640;
    const int height = 480;

    LOG_INFO("test_display_dummy: opening NvSci modules\n");

    NvSciBufModule bufModule = nullptr;
    NvSciError sciErr = NvSciBufModuleOpen(&bufModule);
    if (sciErr != NvSciError_Success) {
        LOG_ERR("NvSciBufModuleOpen failed: %u\n", sciErr);
        return -1;
    }

    NvSciSyncModule syncModule = nullptr;
    sciErr = NvSciSyncModuleOpen(&syncModule);
    if (sciErr != NvSciError_Success) {
        LOG_ERR("NvSciSyncModuleOpen failed: %u\n", sciErr);
        NvSciBufModuleClose(bufModule);
        return -1;
    }

    LOG_INFO("test_display_dummy: initializing compositor (1 display)\n");
    CComposite composite;
    // soloInput=true: this is the only source feeding the display, no real
    // camera pipelines competing for buffers/groups.
    SIPLStatus status = composite.Init(/*uNumDisplays=*/1U,
                                        /*pRect=*/nullptr,
                                        bufModule,
                                        syncModule,
                                        /*soloInput=*/true);
    if (status != NVSIPL_STATUS_OK) {
        LOG_ERR("CComposite::Init failed: %u\n", status);
        NvSciSyncModuleClose(syncModule);
        NvSciBufModuleClose(bufModule);
        return -1;
    }

    LOG_INFO("test_display_dummy: registering synthetic event source\n");
    CEventCompositorSource eventSource;
    // group 0 == PORT-A, mod/out 0 - only choice that matters for a lone
    // synthetic source; see CComposite::PrintDisplayableGroups().
    status = eventSource.Init(&composite, bufModule,
                               /*grpIndex=*/0U, /*modIndex=*/0U, /*outIndex=*/0U,
                               width, height);
    if (status != NVSIPL_STATUS_OK) {
        LOG_ERR("CEventCompositorSource::Init failed: %u\n", status);
        composite.Deinit();
        NvSciSyncModuleClose(syncModule);
        NvSciBufModuleClose(bufModule);
        return -1;
    }

    LOG_INFO("test_display_dummy: starting compositor\n");
    status = composite.Start();
    if (status != NVSIPL_STATUS_OK) {
        LOG_ERR("CComposite::Start failed: %u\n", status);
        eventSource.Deinit();
        composite.Deinit();
        NvSciSyncModuleClose(syncModule);
        NvSciBufModuleClose(bufModule);
        return -1;
    }
    // NOTE: Post() is a silent no-op until Start() has run (m_bRunning
    // check in CComposite::Post) - Start() must happen before PostFrame().

    LOG_INFO("test_display_dummy: posting dummy frames, Ctrl+C to quit\n");
    int phase = 0;
    int frameCount = 0;
    while (!bQuit) {
        std::vector<uint8_t> frame = MakeTestFrame(width, height, phase);
        status = eventSource.PostFrame(frame);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("PostFrame failed: %u\n", status);
            break;
        }
        phase = (phase + 4) % width;
        frameCount++;
        if ((frameCount % 30) == 0) {
            LOG_INFO("test_display_dummy: posted %d frames\n", frameCount);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30fps
    }

    LOG_INFO("test_display_dummy: shutting down\n");
    eventSource.Deinit();
    composite.Stop();
    composite.Deinit();
    NvSciSyncModuleClose(syncModule);
    NvSciBufModuleClose(bufModule);

    return 0;
}
