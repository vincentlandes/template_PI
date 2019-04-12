﻿using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;
// using OpenTK.Input.Mouse;

namespace Template
{
    internal static class CursorPosition
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct POINT
        {
            public int X;
            public int Y;
            public static implicit operator Point(POINT point) { return new Point(point.X, point.Y); }
        }
        [DllImport("user32.dll")]
        public static extern bool GetCursorPos(out POINT lpPoint);
        public static Point GetCursorPosition()
        {
            POINT lpPoint;
            GetCursorPos(out lpPoint);
            return lpPoint;
        }
    }
    public class OpenTKApp : GameWindow
    {
        static int screenID, mousex, mousey;
        static bool mouseLButton = false;
        static Game game;
        protected override void OnLoad( EventArgs e )
        {
            // called upon app init
            GL.ClearColor( Color.Black );
            GL.Enable( EnableCap.Texture2D );
            GL.Hint( HintTarget.PerspectiveCorrectionHint, HintMode.Nicest );
            Width = 512;
            Height = 512;
            game = new Game();
            game.screen = new Surface( Width, Height );
            Sprite.target = game.screen;
            screenID = game.screen.GenTexture();
            game.Init();
        }
        protected override void OnUnload(EventArgs e)
        {
            // called upon app close
            GL.DeleteTextures( 1, ref screenID );
        }
        protected override void OnResize(EventArgs e)
        {
            // called upon window resize
            GL.Viewport(0, 0, Width, Height);
            GL.MatrixMode( MatrixMode.Projection );
            GL.LoadIdentity();
            GL.Ortho( -1.0, 1.0, -1.0, 1.0, 0.0, 4.0 );
        }
        protected override void OnUpdateFrame(FrameEventArgs e)
        {
            // called once per frame; app logic
            var keyboard = OpenTK.Input.Keyboard.GetState();
            if (keyboard[OpenTK.Input.Key.Escape]) this.Exit();
            var mouse = OpenTK.Input.Mouse.GetState();
            Point p = CursorPosition.GetCursorPosition();
            game.SetMouseState( p.X, p.Y, mouse.LeftButton == ButtonState.Pressed );
        }
        protected override void OnRenderFrame(FrameEventArgs e)
        {
            // called once per frame; render
            game.Tick();
            GL.BindTexture( TextureTarget.Texture2D, screenID );
            GL.TexImage2D( TextureTarget.Texture2D,
                           0,
                           PixelInternalFormat.Rgba,
                           game.screen.width,
                           game.screen.height,
                           0,
                           OpenTK.Graphics.OpenGL.PixelFormat.Bgra,
                           PixelType.UnsignedByte,
                           game.screen.pixels
                         );
            GL.Clear( ClearBufferMask.ColorBufferBit );
            GL.MatrixMode( MatrixMode.Modelview );
            GL.LoadIdentity();
            GL.BindTexture( TextureTarget.Texture2D, screenID );
            GL.Begin( PrimitiveType.Quads );
            GL.TexCoord2( 0.0f, 1.0f ); GL.Vertex2( -1.0f, -1.0f );
            GL.TexCoord2( 1.0f, 1.0f ); GL.Vertex2(  1.0f, -1.0f );
            GL.TexCoord2( 1.0f, 0.0f ); GL.Vertex2(  1.0f,  1.0f );
            GL.TexCoord2( 0.0f, 0.0f ); GL.Vertex2( -1.0f,  1.0f );
            GL.End();
            SwapBuffers();
        }
        [STAThread]
        public static void Main()
        {
            // entry point
            using (OpenTKApp app = new OpenTKApp())
            {
                app.Run( 60.0, 0.0 );
            }
        }
    }
}

