/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
package com.xrmech.habla.audio

/**
 * Kotlin port of fdlibm `s_erf.c` (erf only), used by [Dynaudnorm]'s gain bound. The JDK has no
 * erf; fdlibm stays within 1 ulp of the scipy/cephes erf the Python oracle uses. `exp` is
 * [StrictMath.exp], which is fdlibm's `__ieee754_exp`.
 */
internal object Erf {
    private const val TINY = 1e-300
    private const val ERX = 8.45062911510467529297e-01
    private const val EFX = 1.28379167095512586316e-01
    private const val EFX8 = 1.02703333676410069053e+00

    // erf on [0, 0.84375]
    private const val PP0 = 1.28379167095512558561e-01
    private const val PP1 = -3.25042107247001499370e-01
    private const val PP2 = -2.84817495755985104766e-02
    private const val PP3 = -5.77027029648944159157e-03
    private const val PP4 = -2.37630166566501626084e-05
    private const val QQ1 = 3.97917223959155352819e-01
    private const val QQ2 = 6.50222499887672944485e-02
    private const val QQ3 = 5.08130628187576562776e-03
    private const val QQ4 = 1.32494738004321644526e-04
    private const val QQ5 = -3.96022827877536812320e-06

    // erf on [0.84375, 1.25]
    private const val PA0 = -2.36211856075265944077e-03
    private const val PA1 = 4.14856118683748331666e-01
    private const val PA2 = -3.72207876035701323847e-01
    private const val PA3 = 3.18346619901161753674e-01
    private const val PA4 = -1.10894694282396677476e-01
    private const val PA5 = 3.54783043256182359371e-02
    private const val PA6 = -2.16637559486879084300e-03
    private const val QA1 = 1.06420880400844228286e-01
    private const val QA2 = 5.40397917702171048937e-01
    private const val QA3 = 7.18286544141962662868e-02
    private const val QA4 = 1.26171219808761642112e-01
    private const val QA5 = 1.36370839120290507362e-02
    private const val QA6 = 1.19844998467991074170e-02

    // erfc on [1.25, 1/0.35]
    private const val RA0 = -9.86494403484714822705e-03
    private const val RA1 = -6.93858572707181764372e-01
    private const val RA2 = -1.05586262253232909814e+01
    private const val RA3 = -6.23753324503260060396e+01
    private const val RA4 = -1.62396669462573470355e+02
    private const val RA5 = -1.84605092906711035994e+02
    private const val RA6 = -8.12874355063065934246e+01
    private const val RA7 = -9.81432934416914548592e+00
    private const val SA1 = 1.96512716674392571292e+01
    private const val SA2 = 1.37657754143519042600e+02
    private const val SA3 = 4.34565877475229228821e+02
    private const val SA4 = 6.45387271733267880336e+02
    private const val SA5 = 4.29008140027567833386e+02
    private const val SA6 = 1.08635005541779435134e+02
    private const val SA7 = 6.57024977031928170135e+00
    private const val SA8 = -6.04244152148580987438e-02

    // erfc on [1/0.35, 28]
    private const val RB0 = -9.86494292470009928597e-03
    private const val RB1 = -7.99283237680523006574e-01
    private const val RB2 = -1.77579549177547519889e+01
    private const val RB3 = -1.60636384855821916062e+02
    private const val RB4 = -6.37566443368389627722e+02
    private const val RB5 = -1.02509513161107724954e+03
    private const val RB6 = -4.83519191608651397019e+02
    private const val SB1 = 3.03380607434824582924e+01
    private const val SB2 = 3.25792512996573918826e+02
    private const val SB3 = 1.53672958608443695994e+03
    private const val SB4 = 3.19985821950859553908e+03
    private const val SB5 = 2.55305040643316442583e+03
    private const val SB6 = 4.74528541206955367215e+02
    private const val SB7 = -2.24409524465858183362e+01

    fun erf(x: Double): Double {
        val hx = (x.toRawBits() ushr 32).toInt()
        val ix = hx and 0x7fffffff
        if (ix >= 0x7ff00000) { // erf(nan) = nan, erf(+-inf) = +-1
            val i = (hx ushr 31) shl 1
            return (1 - i).toDouble() + 1.0 / x
        }
        if (ix < 0x3feb0000) { // |x| < 0.84375
            if (ix < 0x3e300000) { // |x| < 2**-28
                if (ix < 0x00800000) return 0.125 * (8.0 * x + EFX8 * x) // avoid underflow
                return x + EFX * x
            }
            val z = x * x
            val r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)))
            val s = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))))
            val y = r / s
            return x + x * y
        }
        if (ix < 0x3ff40000) { // 0.84375 <= |x| < 1.25
            val s = Math.abs(x) - 1.0
            val p = PA0 + s * (PA1 + s * (PA2 + s * (PA3 + s * (PA4 + s * (PA5 + s * PA6)))))
            val q = 1.0 + s * (QA1 + s * (QA2 + s * (QA3 + s * (QA4 + s * (QA5 + s * QA6)))))
            return if (hx >= 0) ERX + p / q else -ERX - p / q
        }
        if (ix >= 0x40180000) { // inf > |x| >= 6
            return if (hx >= 0) 1.0 - TINY else TINY - 1.0
        }
        val ax = Math.abs(x)
        val s = 1.0 / (ax * ax)
        val r: Double
        val ss: Double
        if (ix < 0x4006DB6E) { // |x| < 1/0.35
            r = RA0 + s * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7))))))
            ss = 1.0 + s * (SA1 + s * (SA2 + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8)))))))
        } else { // |x| >= 1/0.35
            r = RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6)))))
            ss = 1.0 + s * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7))))))
        }
        val z = Double.fromBits(ax.toRawBits() and -0x100000000L) // low word cleared
        val e = StrictMath.exp(-z * z - 0.5625) * StrictMath.exp((z - ax) * (z + ax) + r / ss)
        return if (hx >= 0) 1.0 - e / ax else e / ax - 1.0
    }
}
