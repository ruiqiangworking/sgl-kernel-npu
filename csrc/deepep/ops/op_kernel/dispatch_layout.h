#ifndef DISPATCH_LAYOUT_H
#define DISPATCH_LAYOUT_H

#include <climits>
#include "kernel_operator.h"

#include "comm_args.h"
#include "data_copy.h"
#include "sync_collectives.h"
#include "moe_distribute_base.h"
#include "dispatch_layout_tiling.h"
namespace MoeDispatchLayout {

constexpr uint32_t UB_32_ALIGN = 32U;
constexpr uint32_t UB_MAX_SIZE = 190U * 1024U;  // 190KB max UB usage per round

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

using namespace AscendC;
using namespace Moe;
template <typename T>
class DispatchLayout
{
public:
    __aicore__ inline DispatchLayout(){};

    __aicore__ inline void Init(GM_ADDR topkIdx, GM_ADDR numTokensPerRank, GM_ADDR numTokensPerExpert,
                                GM_ADDR isTokenInRank, GM_ADDR notifySendData, GM_ADDR sendTokenIdxSmall,
                                GM_ADDR workspace, TPipe *pipe, const DispatchLayoutTilingData *tilingData)
    {
        numTokens_ = tilingData->dispatchLayoutInfo.numTokens;
        numRanks_ = tilingData->dispatchLayoutInfo.numRanks;
        numExperts_ = tilingData->dispatchLayoutInfo.numExperts;
        numTopk_ = tilingData->dispatchLayoutInfo.numTopk;
        tpipe_ = pipe;

        coreIdx_ = GetBlockIdx();
        uint32_t maxAivNum = GetBlockNum();
        aivNum_ = numTokens_ <= maxAivNum ? numTokens_ : maxAivNum;
        if (coreIdx_ >= aivNum_) {
            return;
        }
        uint32_t temp = numTokens_ / aivNum_;
        uint32_t restNum = numTokens_ % aivNum_;
        int64_t topkIdxOffset;
        int64_t isTokenOffset;
        tempTokens_ = temp;
        if (coreIdx_ < restNum) {
            tempTokens_++;
        }
        numTokensPerRank32AlignIntLen_ = Ceil(numRanks_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;
        numTokensPerExpert32AlignIntLen_ = Ceil(numExperts_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;
        if (coreIdx_ < restNum) {
            topkIdxOffset = coreIdx_ * tempTokens_ * numTopk_ * sizeof(int64_t);
            isTokenOffset = coreIdx_ * tempTokens_ * numRanks_ * sizeof(T);
        } else {
            topkIdxOffset = (restNum + coreIdx_ * tempTokens_) * numTopk_ * sizeof(int64_t);
            isTokenOffset = (restNum + coreIdx_ * tempTokens_) * numRanks_ * sizeof(T);
        }
        tempExpertGM_.SetGlobalBuffer((__gm__ T *)notifySendData);
        topkIdxGM_.SetGlobalBuffer((__gm__ int64_t *)(topkIdx + topkIdxOffset));
        numTokensPerRankGM_.SetGlobalBuffer((__gm__ T *)numTokensPerRank);
        numTokensPerExpertGM_.SetGlobalBuffer((__gm__ T *)numTokensPerExpert);
        isTokenInRankGM_.SetGlobalBuffer((__gm__ T *)(isTokenInRank + isTokenOffset));
        sendTokenIdxSmallGM_.SetGlobalBuffer((__gm__ T *)(sendTokenIdxSmall + topkIdxOffset / 2));
    }

    __aicore__ inline uint32_t CalcTokensPerRound()
    {
        // Calculate fixed buffer sizes (independent of token count)
        uint32_t fixedSize = numTokensPerRank32AlignIntLen_ + numTokensPerExpert32AlignIntLen_ +
                            Ceil(numRanks_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;  // seenRankBuf

        // Per-token buffer sizes:
        // topkIdxBuf: numTopk_ * sizeof(int64_t) per token
        // isTokenInRankBuf: numRanks_ * sizeof(T) per token
        // sendTokenIdxSmallBuf: numTopk_ * sizeof(T) per token
        uint32_t perTokenSize = Ceil(numTopk_ * sizeof(int64_t), UB_32_ALIGN) * UB_32_ALIGN +
                               Ceil(numRanks_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN +
                               Ceil(numTopk_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;

        uint32_t availableSize = UB_MAX_SIZE - fixedSize;
        uint32_t tokensPerRound = availableSize / perTokenSize;

        // Ensure at least 1 token per round
        return tokensPerRound > 0 ? tokensPerRound : 1;
    }

    __aicore__ inline void Process()
    {
        if (coreIdx_ >= aivNum_) {
            SyncAll<true>();
            return;
        }

        uint32_t tokensPerRound = CalcTokensPerRound();
        uint32_t numRounds = (tempTokens_ + tokensPerRound - 1) / tokensPerRound;
        int experts_per_rank = numExperts_ / numRanks_;

        // Phase 1: Process tokens in rounds and accumulate counts
        for (uint32_t round = 0; round < numRounds; ++round) {
            uint32_t roundStart = round * tokensPerRound;
            uint32_t roundTokens = (round == numRounds - 1) ? (tempTokens_ - roundStart) : tokensPerRound;

            // Calculate buffer sizes for this round
            uint32_t roundTopkIdxLen = Ceil(roundTokens * numTopk_ * sizeof(int64_t), UB_32_ALIGN) * UB_32_ALIGN;
            uint32_t roundIsTokenInRankLen = Ceil(roundTokens * numRanks_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;

            tpipe_->Reset();
            tpipe_->InitBuffer(topkIdxBuf_, roundTopkIdxLen);
            tpipe_->InitBuffer(numTokensPerRankBuf_, numTokensPerRank32AlignIntLen_);
            tpipe_->InitBuffer(numTokensPerExpertBuf_, numTokensPerExpert32AlignIntLen_);
            tpipe_->InitBuffer(isTokenInRankBuf_, roundIsTokenInRankLen);
            tpipe_->InitBuffer(seenRankBuf_, Ceil(numRanks_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN);

            // Load topkIdx for this round
            LocalTensor<int64_t> topkIdxTensor = topkIdxBuf_.AllocTensor<int64_t>();
            const DataCopyExtParams dataCopyParams{1U, roundTopkIdxLen, 0U, 0U, 0U};
            const DataCopyPadExtParams<int64_t> padParams{false, 0U, 0U, 0U};
            DataCopyPad(topkIdxTensor, topkIdxGM_[roundStart * numTopk_], dataCopyParams, padParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();

            LocalTensor<T> numTokensPerRankTensor = numTokensPerRankBuf_.AllocTensor<T>();
            LocalTensor<T> numTokensPerExpertTensor = numTokensPerExpertBuf_.AllocTensor<T>();
            LocalTensor<T> isTokenInRankTensor = isTokenInRankBuf_.AllocTensor<T>();
            LocalTensor<T> seenRankTensor = seenRankBuf_.AllocTensor<T>();

            Duplicate<T>(numTokensPerRankTensor, 0, numRanks_);
            Duplicate<T>(numTokensPerExpertTensor, 0, numExperts_);
            Duplicate<T>(isTokenInRankTensor, 0, roundTokens * numRanks_);
            SyncFunc<AscendC::HardEvent::V_S>();

            // Process tokens in this round
            for (uint32_t i = 0; i < roundTokens; ++i) {
                SyncFunc<AscendC::HardEvent::S_V>();
                Duplicate<T>(seenRankTensor, 0, numRanks_);
                SyncFunc<AscendC::HardEvent::V_S>();
                for (uint32_t j = 0; j < numTopk_; ++j) {
                    int64_t expert_idx = topkIdxTensor.GetValue(i * numTopk_ + j);
                    uint32_t per_expert_num = numTokensPerExpertTensor.GetValue(expert_idx) + 1;
                    numTokensPerExpertTensor.SetValue(expert_idx, per_expert_num);
                    int rank_id = expert_idx / experts_per_rank;
                    if (!seenRankTensor.GetValue(rank_id)) {
                        uint32_t per_rank_num = numTokensPerRankTensor.GetValue(rank_id) + 1;
                        isTokenInRankTensor.SetValue(i * numRanks_ + rank_id, 1);
                        seenRankTensor.SetValue(rank_id, 1);
                        numTokensPerRankTensor.SetValue(rank_id, per_rank_num);
                    }
                }
            }

            // Write isTokenInRank for this round
            uint32_t sendSize = roundTokens * numRanks_ * sizeof(T);
            const DataCopyExtParams isTokenInRankDataCopyParams{1U, sendSize, 0U, 0U, 0U};
            DataCopyPad(isTokenInRankGM_[roundStart * numRanks_], isTokenInRankTensor, isTokenInRankDataCopyParams);

            // Atomic add for accumulated counts
            AscendC::SetAtomicAdd<T>();
            const DataCopyExtParams tempExpertDataCopyParams{1U, numTokensPerExpert32AlignIntLen_, 0U, 0U, 0U};
            for (uint32_t i = coreIdx_ + 1; i < aivNum_; ++i) {
                DataCopyPad(tempExpertGM_[i * numExperts_], numTokensPerExpertTensor, tempExpertDataCopyParams);
            }
            sendSize = numRanks_ * sizeof(T);
            const DataCopyExtParams numTokensPerRankDataCopyParams{1U, sendSize, 0U, 0U, 0U};
            DataCopyPad(numTokensPerRankGM_, numTokensPerRankTensor, numTokensPerRankDataCopyParams);
            sendSize = numExperts_ * sizeof(T);
            const DataCopyExtParams numTokensPerExpertDataCopyParams{1U, sendSize, 0U, 0U, 0U};
            DataCopyPad(numTokensPerExpertGM_, numTokensPerExpertTensor, numTokensPerExpertDataCopyParams);
            AscendC::SetAtomicNone();
            PipeBarrier<PIPE_MTE3>();
        }

        // Sync all cores after phase 1
        SyncAll<true>();

        // Phase 2: Calculate sendTokenIdxSmall in rounds
        for (uint32_t round = 0; round < numRounds; ++round) {
            uint32_t roundStart = round * tokensPerRound;
            uint32_t roundTokens = (round == numRounds - 1) ? (tempTokens_ - roundStart) : tokensPerRound;

            // Calculate buffer sizes for this round
            uint32_t roundTopkIdxLen = Ceil(roundTokens * numTopk_ * sizeof(int64_t), UB_32_ALIGN) * UB_32_ALIGN;
            uint32_t roundSendTokenIdxLen = Ceil(roundTokens * numTopk_ * sizeof(T), UB_32_ALIGN) * UB_32_ALIGN;

            tpipe_->Reset();
            tpipe_->InitBuffer(topkIdxBuf_, roundTopkIdxLen);
            tpipe_->InitBuffer(numTokensPerExpertBuf_, numTokensPerExpert32AlignIntLen_);
            tpipe_->InitBuffer(sendTokenIdxSmallBuf_, roundSendTokenIdxLen);

            // Load topkIdx for this round
            LocalTensor<int64_t> topkIdxTensor = topkIdxBuf_.AllocTensor<int64_t>();
            const DataCopyExtParams dataCopyParams{1U, roundTopkIdxLen, 0U, 0U, 0U};
            const DataCopyPadExtParams<int64_t> padParams{false, 0U, 0U, 0U};
            DataCopyPad(topkIdxTensor, topkIdxGM_[roundStart * numTopk_], dataCopyParams, padParams);

            // Load accumulated numTokensPerExpert
            LocalTensor<T> numTokensPerExpertTensor = numTokensPerExpertBuf_.AllocTensor<T>();
            const DataCopyExtParams tempExpertDataCopyParams{1U, numTokensPerExpert32AlignIntLen_, 0U, 0U, 0U};
            const DataCopyPadExtParams<T> tempPadParams{false, 0U, 0U, 0U};
            DataCopyPad(numTokensPerExpertTensor, tempExpertGM_[coreIdx_ * numExperts_], tempExpertDataCopyParams,
                        tempPadParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();

            LocalTensor<T> sendTokenIdxSmallTensor = sendTokenIdxSmallBuf_.AllocTensor<T>();

            // Calculate sendTokenIdxSmall for this round
            for (uint32_t i = 0; i < roundTokens; ++i) {
                for (uint32_t j = 0; j < numTopk_; ++j) {
                    int64_t expert_idx = topkIdxTensor.GetValue(i * numTopk_ + j);
                    T valT = numTokensPerExpertTensor(expert_idx);
                    sendTokenIdxSmallTensor(i * numTopk_ + j) = valT;
                    numTokensPerExpertTensor(expert_idx) = valT + 1;
                }
            }

            // Write back updated numTokensPerExpert for next round
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            DataCopyPad(tempExpertGM_[coreIdx_ * numExperts_], numTokensPerExpertTensor, tempExpertDataCopyParams);

            // Write sendTokenIdxSmall for this round
            const DataCopyExtParams sendTokenIdxSmallDataCopyParams{
                1U, static_cast<uint32_t>(roundTokens * numTopk_ * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(sendTokenIdxSmallGM_[roundStart * numTopk_], sendTokenIdxSmallTensor,
                        sendTokenIdxSmallDataCopyParams);
            PipeBarrier<PIPE_MTE3>();
        }
    }

private:
    GlobalTensor<int64_t> topkIdxGM_;
    GlobalTensor<T> numTokensPerRankGM_;
    GlobalTensor<T> numTokensPerExpertGM_;
    GlobalTensor<T> isTokenInRankGM_;
    GlobalTensor<T> tempExpertGM_;
    GlobalTensor<T> sendTokenIdxSmallGM_;

    TBuf<> topkIdxBuf_;
    TBuf<> numTokensPerRankBuf_;
    TBuf<> numTokensPerExpertBuf_;
    TBuf<> isTokenInRankBuf_;
    TBuf<> seenRankBuf_;
    TBuf<> sendTokenIdxSmallBuf_;

    TPipe *tpipe_{nullptr};
    uint32_t numTokens_{0};
    uint32_t numRanks_{0};
    uint32_t numExperts_{0};
    uint32_t numTopk_{0};
    uint32_t coreIdx_{0};
    uint32_t aivNum_{0};
    uint32_t tempTokens_{0};

    uint32_t numTokensPerRank32AlignIntLen_{0};
    uint32_t numTokensPerExpert32AlignIntLen_{0};
};
}  // namespace MoeDispatchLayout

#endif  // DISPATCH_LAYOUT_H
