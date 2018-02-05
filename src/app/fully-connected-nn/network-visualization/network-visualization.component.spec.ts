import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { NetworkVisualizationComponent } from './network-visualization.component';

describe('NetworkVisualizationComponent', () => {
  let component: NetworkVisualizationComponent;
  let fixture: ComponentFixture<NetworkVisualizationComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ NetworkVisualizationComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(NetworkVisualizationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
