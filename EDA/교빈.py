            

            # ui.card(
            #     ui.card_header("요금 중심 마인드맵"),
            #     ui.layout_columns(
            #         # ────── 좌측: Mermaid 마인드맵 ──────
            #         ui.HTML("""
            #         <div style="padding: 16px;">
            #             <div class="mermaid" style="font-size: 30px;">
            #             flowchart TD
            #                 D["지상 무효전력량(kVarh)"] --> Q(("Q: 무효전력량(kVarh)"))
            #                 E["진상무효전력량(kVarh)"] --> Q

            #                 Q --> F[지상/진상 역률]
            #                 B(["P: 전력사용량(kWh)"]) --> F["지상/진상 역률(%)"]
            #                 F -.->|역률에 따른 추가 요금 부과|A[전기요금]

            #                 B -->|회귀계수: 107.25| A["전기요금(원)"]
            #                 B --> C["탄소배출량(tCO2)"]
            #                 C --> A
            #             </div>
            #         </div>

            #         <script type="module">
            #         import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            #         mermaid.initialize({ startOnLoad: true });
            #         </script>
            #         """),

            #         # ────── 우측: 설명 ──────
            #         ui.HTML("""
            #         <div style="font-size: 16px; padding: 16px;">
            #             <br><br><br>
            #             <strong>전력 관계식</strong>
            #             <ul>
            #             <li><strong>피상전력 관계식:</strong> S² = P² + Q²  
            #                 피상전력(S)은 유효전력(P)과 무효전력(Q)의 벡터 합으로, 전기설비가 실제로 부담하는 전체 전력량을 나타냅니다.</li><br>
                        
            #             <li><strong>역률(Power Factor):</strong> 역률 = P / S  
            #                 유효전력이 전체 피상전력에서 차지하는 비율로, 1에 가까울수록 전력 사용이 효율적입니다.  
            #                 역률이 낮을수록 무효전력 비중이 높아져, 산업용 설비에서는 벌금 또는 기본요금 증가로 이어질 수 있습니다.</li><br>
                        
            #             <li><strong>지상과 진상은 동시에 성립하지 않음:</strong>  
            #                 지상무효전력은 유도성 부하에서, 진상무효전력은 용량성 부하에서 발생하므로  
            #                 특정 시점에는 두 중 하나만 발생합니다. 전류가 전압보다 늦을 때는 지상, 빠를 때는 진상 상태입니다.</li><br>
            #             </ul>
            #         </div>
            #         """),

            #         col_widths=[6, 6]
            #     )
            #             ),

